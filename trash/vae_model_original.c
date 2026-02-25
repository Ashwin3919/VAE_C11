/*
 * vae_model.c — Conditional VAE for MNIST in plain C
 *
 * Three progressive model sizes (select at compile time):
 *   v1:  784->256->128->32->128->256->784   (digits 0-1, ~370K params)
 *   v2:  784->512->256->64->256->512->784   (digits 0-1, ~1.1M params)
 *   v3:  784->640->320->128->320->640->784  (digits 0-9, ~1.7M params)
 *
 * Conditioning: one-hot digit label is appended to the encoder input
 * and prepended to the decoder input (standard CVAE). This lets you
 * generate any specific digit after training.
 *
 * Build:
 *   make           => v1, binary digits
 *   make mid       => v2, binary digits
 *   make full      => v3, all digits
 *   make debug     => v1 with -g
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Compile-time config                                                  */
/* ------------------------------------------------------------------ */
#ifdef V3
#define VERSION "v3"
#define H1 640
#define H2 320
#define LATENT 128
#define NUM_CLASSES 10
#define FULL_MNIST 1
#define EPOCHS 800
#define LR 0.0008f
#elif defined(V2)
#define VERSION "v2"
#define H1 512
#define H2 256
#define LATENT 64
#define NUM_CLASSES 2
#define EPOCHS 400
#define LR 0.001f
#else
#define VERSION "v1"
#define H1 256
#define H2 128
#define LATENT 32
#define NUM_CLASSES 2
#define EPOCHS 300
#define LR 0.001f
#endif

#define IMAGE_SIZE 784
/* encoder input = image pixels + one-hot label */
#define ENC_IN (IMAGE_SIZE + NUM_CLASSES)
/* decoder input = latent vector + one-hot label */
#define DEC_IN (LATENT + NUM_CLASSES)

#define BATCH_SIZE 64
#define BETA_START 0.00001f
#define BETA_END 0.0005f
#define BETA_WARMUP 100
#define BETA_ANNEAL 150
#define GRAD_CLIP 5.0f
#define SAVE_EVERY 50

#define RESULT_DIR "results_main/" VERSION
#define MODEL_DIR "models"
#define MODEL_FILE MODEL_DIR "/vae_" VERSION ".bin"

/* ------------------------------------------------------------------ */
/* Data                                                                 */
/* ------------------------------------------------------------------ */
typedef struct {
  float *_mem; /* single slab — every pointer below is an offset into it */

  /* weights */
  float *enc_w1, *enc_b1; /* ENC_IN  -> H1        */
  float *enc_w2, *enc_b2; /* H1      -> H2        */
  float *mu_w, *mu_b;     /* H2      -> LATENT    */
  float *lv_w, *lv_b;     /* H2      -> LATENT    */
  float *dec_w1, *dec_b1; /* DEC_IN  -> H1        */
  float *dec_w2, *dec_b2; /* H1      -> H2        */
  float *dec_w3, *dec_b3; /* H2      -> IMAGE_SIZE */

  /* activations — [BATCH_SIZE * layer_width] */
  float *enc_h1, *enc_h2;
  float *mu, *logvar, *z;
  float *dec_h1, *dec_h2, *output;

  /* concatenated encoder/decoder inputs — [BATCH_SIZE * dim] */
  float *enc_in_buf; /* [BATCH_SIZE, ENC_IN] */
  float *dec_in_buf; /* [BATCH_SIZE, DEC_IN] */

  /* pre-activation buffers — [BATCH_SIZE * layer_width] */
  float *pre_eh1, *pre_eh2;
  float *pre_dh1, *pre_dh2, *pre_out;
  float *eps_buf;

  /* gradient accumulators (reset each batch, weight-shaped) */
  float *dw3, *db3;
  float *dw2, *db2;
  float *dw1, *db1;
  float *d_muw, *d_mub;
  float *d_lvw, *d_lvb;
  float *d_ew2, *d_eb2;
  float *d_ew1, *d_eb1;

  /* Adam moment buffers */
  float *m_ew1, *v_ew1, *m_eb1, *v_eb1;
  float *m_ew2, *v_ew2, *m_eb2, *v_eb2;
  float *m_muw, *v_muw, *m_mub, *v_mub;
  float *m_lvw, *v_lvw, *m_lvb, *v_lvb;
  float *m_dw1, *v_dw1, *m_db1, *v_db1;
  float *m_dw2, *v_dw2, *m_db2, *v_db2;
  float *m_dw3, *v_dw3, *m_db3, *v_db3;

  int adam_t;
} VAE;

typedef struct {
  float **images;
  int *labels;
  int count;
} Dataset;

/* ------------------------------------------------------------------ */
/* RNG                                                                  */
/* ------------------------------------------------------------------ */
static uint64_t rng = 0x123456789ABCULL;

static inline float randU(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (float)(rng & 0xFFFFFF) / (float)0x1000000; /* [0, 1) */
}

/* uniform integer in [0, n-1] using the same xorshift RNG */
static inline int randi(int n) { return (int)(randU() * (float)n) % n; }

static float randn(void) {
  static int spare_ready;
  static float spare;
  if (spare_ready) {
    spare_ready = 0;
    return spare;
  }
  float u, v, s;
  do {
    u = randU() * 2 - 1;
    v = randU() * 2 - 1;
    s = u * u + v * v;
  } while (s >= 1 || s == 0);
  s = sqrtf(-2.0f * logf(s) / s);
  spare = v * s;
  spare_ready = 1;
  return u * s;
}

/* ------------------------------------------------------------------ */
/* Math helpers                                                         */
/* ------------------------------------------------------------------ */
static inline float clamp(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
static inline float elu(float x) {
  return x > 0.0f ? x : 0.2f * (expf(clamp(x, -15, 0)) - 1);
}
static inline float elu_d(float pre) { /* d(elu)/d(pre-activation) */
  return pre > 0.0f ? 1.0f : 0.2f * expf(clamp(pre, -15, 0));
}
static inline float sigmoid(float x) {
  x = clamp(x, -15, 15);
  return 1.0f / (1.0f + expf(-x));
}
static float clip(float g) {
  if (!isfinite(g))
    return 0.0f;
  return clamp(g, -GRAD_CLIP, GRAD_CLIP);
}
static void he_init(float *w, int in, int out) {
  float s = sqrtf(2.0f / in);
  for (int i = 0; i < in * out; i++)
    w[i] = randn() * s;
}
/* linear: out[out_n] = W[in_n*out_n]*in[in_n] + b[out_n] */
static void linear(float *out, const float *in, const float *w, const float *b,
                   int in_n, int out_n) {
  for (int j = 0; j < out_n; j++)
    out[j] = b[j];
  for (int i = 0; i < in_n; i++) {
    float x = in[i];
    const float *wr = w + i * out_n;
    for (int j = 0; j < out_n; j++)
      out[j] += x * wr[j];
  }
}

/*
 * linear_batch: true batched matmul
 *   Y[bsz, out_n] = X[bsz, in_n] * W[in_n, out_n] + b[out_n]
 * All bsz samples are processed together through a single W, enabling
 * cache-friendly access across samples (true batch matmul, not accumulation).
 */
static void linear_batch(float *Y, const float *X, const float *W,
                         const float *b, int bsz, int in_n, int out_n) {
  for (int s = 0; s < bsz; s++) {
    float *y = Y + s * out_n;
    for (int j = 0; j < out_n; j++)
      y[j] = b[j];
  }
  for (int s = 0; s < bsz; s++) {
    const float *x = X + s * in_n;
    float *y = Y + s * out_n;
    for (int i = 0; i < in_n; i++) {
      float xi = x[i];
      const float *wr = W + i * out_n;
      for (int j = 0; j < out_n; j++)
        y[j] += xi * wr[j];
    }
  }
}

/* ------------------------------------------------------------------ */
/* Adam                                                                 */
/* ------------------------------------------------------------------ */
static void adam_update(float *w, float *dw, float *m, float *v, int n,
                        float lr, int t) {
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
  for (int i = 0; i < n; i++) {
    float g = clip(dw[i]);
    m[i] = b1 * m[i] + (1 - b1) * g;
    v[i] = b2 * v[i] + (1 - b2) * g * g;
    w[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
    dw[i] = 0.0f;
  }
}

/* ------------------------------------------------------------------ */
/* Alloc / free                                                         */
/* ------------------------------------------------------------------ */
VAE *create_vae(void) {
  VAE *m = calloc(1, sizeof(VAE));
  if (!m) {
    fprintf(stderr, "[FATAL] calloc VAE struct failed\n");
    exit(1);
  }

  /* One contiguous slab for all float arrays. */
#define BS BATCH_SIZE
  size_t n = 0;
  /* weights */
  n += (size_t)ENC_IN * H1 + H1 + H1 * H2 + H2 +
       2 * ((size_t)H2 * LATENT + LATENT) + (size_t)DEC_IN * H1 + H1 + H1 * H2 +
       H2 + (size_t)H2 * IMAGE_SIZE + IMAGE_SIZE;
  /* activations (batch-sized rows) */
  n += (size_t)BS * (H1 + H2 + 3 * LATENT + H1 + H2 + IMAGE_SIZE);
  /* input concat buffers */
  n += (size_t)BS * (ENC_IN + DEC_IN);
  /* pre-activation buffers */
  n += (size_t)BS * (H1 + H2 + H1 + H2 + IMAGE_SIZE + LATENT);
  /* gradient accumulators */
  n += (size_t)H2 * IMAGE_SIZE + IMAGE_SIZE + H1 * H2 + H2 +
       (size_t)DEC_IN * H1 + H1 + 2 * ((size_t)H2 * LATENT + LATENT) + H1 * H2 +
       H2 + (size_t)ENC_IN * H1 + H1;
  /* Adam moment buffers (2x each weight/bias) */
  n += 2 * ((size_t)ENC_IN * H1 + H1 + H1 * H2 + H2 +
            2 * ((size_t)H2 * LATENT + LATENT) + (size_t)DEC_IN * H1 + H1 +
            H1 * H2 + H2 + (size_t)H2 * IMAGE_SIZE + IMAGE_SIZE);
#undef BS

  m->_mem = calloc(n, sizeof(float));
  if (!m->_mem) {
    fprintf(stderr, "[FATAL] calloc VAE slab failed\n");
    exit(1);
  }

  float *p = m->_mem;
#define NEXT(ptr, sz) ((ptr) = p, p += (sz))

  /* weights */
  NEXT(m->enc_w1, ENC_IN * H1);
  NEXT(m->enc_b1, H1);
  NEXT(m->enc_w2, H1 * H2);
  NEXT(m->enc_b2, H2);
  NEXT(m->mu_w, H2 * LATENT);
  NEXT(m->mu_b, LATENT);
  NEXT(m->lv_w, H2 * LATENT);
  NEXT(m->lv_b, LATENT);
  NEXT(m->dec_w1, DEC_IN * H1);
  NEXT(m->dec_b1, H1);
  NEXT(m->dec_w2, H1 * H2);
  NEXT(m->dec_b2, H2);
  NEXT(m->dec_w3, H2 * IMAGE_SIZE);
  NEXT(m->dec_b3, IMAGE_SIZE);
  /* activations */
  NEXT(m->enc_h1, BATCH_SIZE * H1);
  NEXT(m->enc_h2, BATCH_SIZE * H2);
  NEXT(m->mu, BATCH_SIZE * LATENT);
  NEXT(m->logvar, BATCH_SIZE * LATENT);
  NEXT(m->z, BATCH_SIZE * LATENT);
  NEXT(m->dec_h1, BATCH_SIZE * H1);
  NEXT(m->dec_h2, BATCH_SIZE * H2);
  NEXT(m->output, BATCH_SIZE * IMAGE_SIZE);
  /* input buffers */
  NEXT(m->enc_in_buf, BATCH_SIZE * ENC_IN);
  NEXT(m->dec_in_buf, BATCH_SIZE * DEC_IN);
  /* pre-activations */
  NEXT(m->pre_eh1, BATCH_SIZE * H1);
  NEXT(m->pre_eh2, BATCH_SIZE * H2);
  NEXT(m->pre_dh1, BATCH_SIZE * H1);
  NEXT(m->pre_dh2, BATCH_SIZE * H2);
  NEXT(m->pre_out, BATCH_SIZE * IMAGE_SIZE);
  NEXT(m->eps_buf, BATCH_SIZE * LATENT);
  /* gradient accumulators */
  NEXT(m->dw3, H2 * IMAGE_SIZE);
  NEXT(m->db3, IMAGE_SIZE);
  NEXT(m->dw2, H1 * H2);
  NEXT(m->db2, H2);
  NEXT(m->dw1, DEC_IN * H1);
  NEXT(m->db1, H1);
  NEXT(m->d_muw, H2 * LATENT);
  NEXT(m->d_mub, LATENT);
  NEXT(m->d_lvw, H2 * LATENT);
  NEXT(m->d_lvb, LATENT);
  NEXT(m->d_ew2, H1 * H2);
  NEXT(m->d_eb2, H2);
  NEXT(m->d_ew1, ENC_IN * H1);
  NEXT(m->d_eb1, H1);
  /* Adam moment buffers */
  NEXT(m->m_ew1, ENC_IN * H1);
  NEXT(m->v_ew1, ENC_IN * H1);
  NEXT(m->m_eb1, H1);
  NEXT(m->v_eb1, H1);
  NEXT(m->m_ew2, H1 * H2);
  NEXT(m->v_ew2, H1 * H2);
  NEXT(m->m_eb2, H2);
  NEXT(m->v_eb2, H2);
  NEXT(m->m_muw, H2 * LATENT);
  NEXT(m->v_muw, H2 * LATENT);
  NEXT(m->m_mub, LATENT);
  NEXT(m->v_mub, LATENT);
  NEXT(m->m_lvw, H2 * LATENT);
  NEXT(m->v_lvw, H2 * LATENT);
  NEXT(m->m_lvb, LATENT);
  NEXT(m->v_lvb, LATENT);
  NEXT(m->m_dw1, DEC_IN * H1);
  NEXT(m->v_dw1, DEC_IN * H1);
  NEXT(m->m_db1, H1);
  NEXT(m->v_db1, H1);
  NEXT(m->m_dw2, H1 * H2);
  NEXT(m->v_dw2, H1 * H2);
  NEXT(m->m_db2, H2);
  NEXT(m->v_db2, H2);
  NEXT(m->m_dw3, H2 * IMAGE_SIZE);
  NEXT(m->v_dw3, H2 * IMAGE_SIZE);
  NEXT(m->m_db3, IMAGE_SIZE);
  NEXT(m->v_db3, IMAGE_SIZE);
#undef NEXT

  he_init(m->enc_w1, ENC_IN, H1);
  he_init(m->enc_w2, H1, H2);
  he_init(m->mu_w, H2, LATENT);
  he_init(m->lv_w, H2, LATENT);
  he_init(m->dec_w1, DEC_IN, H1);
  he_init(m->dec_w2, H1, H2);
  he_init(m->dec_w3, H2, IMAGE_SIZE);
  for (int i = 0; i < H2 * LATENT; i++)
    m->lv_w[i] *= 0.01f;

  m->adam_t = 0;
  long pc = (long)ENC_IN * H1 + H1 + H1 * H2 + H2 +
            2 * ((long)H2 * LATENT + LATENT) + (long)DEC_IN * H1 + H1 +
            H1 * H2 + H2 + (long)H2 * IMAGE_SIZE + IMAGE_SIZE;
  printf("[INFO] CVAE %s: enc_in=%d  latent=%d  classes=%d  params=%ldK\n",
         VERSION, ENC_IN, LATENT, NUM_CLASSES, pc / 1000);
  return m;
}

void free_vae(VAE *m) {
  if (!m)
    return;
  free(m->_mem); /* single free releases the entire slab */
  free(m);
}

/* ------------------------------------------------------------------ */
/* Forward: label is the digit class [0, NUM_CLASSES)                  */
/* ------------------------------------------------------------------ */

/* build one-hot vector of length NUM_CLASSES */
static void one_hot(float *out, int cls) {
  memset(out, 0, NUM_CLASSES * sizeof(float));
  if (cls >= 0 && cls < NUM_CLASSES)
    out[cls] = 1.0f;
}

/*
 * vae_forward — true batched forward pass.
 * xs[bsz]: array of pointers to IMAGE_SIZE-float image arrays.
 * labels[bsz]: digit class for each sample.
 * All activations stored in contiguous m-> slab rows: m->enc_h1[s*H1 ..
 * (s+1)*H1).
 */
void vae_forward(VAE *m, float **xs, const int *labels, int bsz, int training) {
  /* Build enc_in_buf[bsz, ENC_IN] */
  for (int s = 0; s < bsz; s++) {
    float *dst = m->enc_in_buf + s * ENC_IN;
    memcpy(dst, xs[s], IMAGE_SIZE * sizeof(float));
    one_hot(dst + IMAGE_SIZE, labels[s]);
  }

  /* Encoder layer 1 — all bsz samples in one matmul */
  linear_batch(m->pre_eh1, m->enc_in_buf, m->enc_w1, m->enc_b1, bsz, ENC_IN,
               H1);
  for (int i = 0; i < bsz * H1; i++)
    m->enc_h1[i] = elu(m->pre_eh1[i]);

  /* Encoder layer 2 */
  linear_batch(m->pre_eh2, m->enc_h1, m->enc_w2, m->enc_b2, bsz, H1, H2);
  for (int i = 0; i < bsz * H2; i++)
    m->enc_h2[i] = elu(m->pre_eh2[i]);

  /* mu and logvar heads */
  linear_batch(m->mu, m->enc_h2, m->mu_w, m->mu_b, bsz, H2, LATENT);
  linear_batch(m->logvar, m->enc_h2, m->lv_w, m->lv_b, bsz, H2, LATENT);
  for (int i = 0; i < bsz * LATENT; i++)
    m->logvar[i] = clamp(m->logvar[i], -10.0f, 4.0f);

  /* Reparametrisation: z[s] = mu[s] + exp(0.5*lv[s]) * eps */
  for (int s = 0; s < bsz; s++) {
    float *mu_s = m->mu + s * LATENT;
    float *lv_s = m->logvar + s * LATENT;
    float *z_s = m->z + s * LATENT;
    float *eps_s = m->eps_buf + s * LATENT;
    for (int i = 0; i < LATENT; i++) {
      float eps = training ? randn() : 0.0f;
      eps_s[i] = eps;
      z_s[i] = mu_s[i] + expf(0.5f * lv_s[i]) * eps;
    }
  }

  /* Build dec_in_buf[bsz, DEC_IN] */
  for (int s = 0; s < bsz; s++) {
    float *dst = m->dec_in_buf + s * DEC_IN;
    memcpy(dst, m->z + s * LATENT, LATENT * sizeof(float));
    one_hot(dst + LATENT, labels[s]);
  }

  /* Decoder layer 1 */
  linear_batch(m->pre_dh1, m->dec_in_buf, m->dec_w1, m->dec_b1, bsz, DEC_IN,
               H1);
  for (int i = 0; i < bsz * H1; i++)
    m->dec_h1[i] = elu(m->pre_dh1[i]);

  /* Decoder layer 2 */
  linear_batch(m->pre_dh2, m->dec_h1, m->dec_w2, m->dec_b2, bsz, H1, H2);
  for (int i = 0; i < bsz * H2; i++)
    m->dec_h2[i] = elu(m->pre_dh2[i]);

  /* Output layer */
  linear_batch(m->pre_out, m->dec_h2, m->dec_w3, m->dec_b3, bsz, H2,
               IMAGE_SIZE);
  for (int i = 0; i < bsz * IMAGE_SIZE; i++)
    m->output[i] = sigmoid(m->pre_out[i]);
}

/* ------------------------------------------------------------------ */
/* Loss (batch average)                                                 */
/* ------------------------------------------------------------------ */
float vae_loss(VAE *m, float **xs, int bsz, float beta) {
  float total = 0.0f;
  for (int s = 0; s < bsz; s++) {
    const float *x = xs[s];
    const float *out = m->output + s * IMAGE_SIZE;
    const float *mu = m->mu + s * LATENT;
    const float *lv = m->logvar + s * LATENT;

    float recon = 0.0f;
    for (int i = 0; i < IMAGE_SIZE; i++) {
      float p = clamp(out[i], 1e-7f, 1 - 1e-7f);
      recon -= x[i] * logf(p) + (1 - x[i]) * logf(1 - p);
    }
    recon /= IMAGE_SIZE;

    float kl = 0.0f;
    for (int i = 0; i < LATENT; i++)
      kl += 1 + lv[i] - mu[i] * mu[i] - expf(lv[i]);
    kl = -0.5f * kl / LATENT;

    float t = recon + beta * kl;
    total += isfinite(t) ? t : 1.0f;
  }
  return total / bsz;
}

/* ------------------------------------------------------------------ */
/* Gradient accumulators live in the VAE slab (no static globals)       */
/* ------------------------------------------------------------------ */
static void reset_grads(VAE *m) {
  memset(m->dw3, 0, H2 * IMAGE_SIZE * sizeof(float));
  memset(m->db3, 0, IMAGE_SIZE * sizeof(float));
  memset(m->dw2, 0, H1 * H2 * sizeof(float));
  memset(m->db2, 0, H2 * sizeof(float));
  memset(m->dw1, 0, DEC_IN * H1 * sizeof(float));
  memset(m->db1, 0, H1 * sizeof(float));
  memset(m->d_muw, 0, H2 * LATENT * sizeof(float));
  memset(m->d_mub, 0, LATENT * sizeof(float));
  memset(m->d_lvw, 0, H2 * LATENT * sizeof(float));
  memset(m->d_lvb, 0, LATENT * sizeof(float));
  memset(m->d_ew2, 0, H1 * H2 * sizeof(float));
  memset(m->d_eb2, 0, H2 * sizeof(float));
  memset(m->d_ew1, 0, ENC_IN * H1 * sizeof(float));
  memset(m->d_eb1, 0, H1 * sizeof(float));
}

/* ------------------------------------------------------------------ */
/* Backward pass — accumulates gradients for all bsz samples            */
/* scale = 1/bsz is applied only to KL terms here; the reconstruction   */
/* gradient through d_z is already batch-normalised from the output      */
/* layer (Fix 1: avoids double-scaling the reconstruction path).         */
/* ------------------------------------------------------------------ */
void vae_backward(VAE *m, float **xs, const int *labels, int bsz, float beta) {
  (void)
      labels; /* labels are embedded in enc_in_buf/dec_in_buf by vae_forward */
  float scale = 1.0f / bsz;

  for (int s = 0; s < bsz; s++) {
    const float *x = xs[s];
    const float *out_s = m->output + s * IMAGE_SIZE;
    const float *dec_h2s = m->dec_h2 + s * H2;
    const float *dec_h1s = m->dec_h1 + s * H1;
    const float *enc_h2s = m->enc_h2 + s * H2;
    const float *enc_h1s = m->enc_h1 + s * H1;
    const float *mu_s = m->mu + s * LATENT;
    const float *lv_s = m->logvar + s * LATENT;
    const float *eps_s = m->eps_buf + s * LATENT;
    const float *pdh2_s = m->pre_dh2 + s * H2;
    const float *pdh1_s = m->pre_dh1 + s * H1;
    const float *peh2_s = m->pre_eh2 + s * H2;
    const float *peh1_s = m->pre_eh1 + s * H1;
    const float *enc_in_s = m->enc_in_buf + s * ENC_IN;
    const float *dec_in_s = m->dec_in_buf + s * DEC_IN;

    /* --- output: BCE+sigmoid combined, d = (out - x) / (IMAGE_SIZE * bsz) ---
     */
    float d_out[IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; i++) {
      d_out[i] = (out_s[i] - x[i]) / ((float)IMAGE_SIZE * bsz);
      m->db3[i] += d_out[i];
    }
    float d_dh2[H2];
    memset(d_dh2, 0, H2 * sizeof(float));
    for (int i = 0; i < H2; i++) {
      for (int j = 0; j < IMAGE_SIZE; j++) {
        m->dw3[i * IMAGE_SIZE + j] += dec_h2s[i] * d_out[j];
        d_dh2[i] += m->dec_w3[i * IMAGE_SIZE + j] * d_out[j];
      }
    }

    /* --- decoder layer 2 --- */
    float d_pdh2[H2];
    for (int i = 0; i < H2; i++)
      d_pdh2[i] = d_dh2[i] * elu_d(pdh2_s[i]);
    float d_dh1[H1];
    memset(d_dh1, 0, H1 * sizeof(float));
    for (int j = 0; j < H2; j++)
      m->db2[j] += d_pdh2[j];
    for (int i = 0; i < H1; i++)
      for (int j = 0; j < H2; j++) {
        m->dw2[i * H2 + j] += dec_h1s[i] * d_pdh2[j];
        d_dh1[i] += m->dec_w2[i * H2 + j] * d_pdh2[j];
      }

    /* --- decoder layer 1 --- */
    float d_pdh1[H1];
    for (int i = 0; i < H1; i++)
      d_pdh1[i] = d_dh1[i] * elu_d(pdh1_s[i]);
    for (int j = 0; j < H1; j++)
      m->db1[j] += d_pdh1[j];

    float d_z[LATENT];
    memset(d_z, 0, LATENT * sizeof(float));
    for (int i = 0; i < DEC_IN; i++)
      for (int j = 0; j < H1; j++) {
        m->dw1[i * H1 + j] += dec_in_s[i] * d_pdh1[j];
        if (i < LATENT)
          d_z[i] += m->dec_w1[i * H1 + j] * d_pdh1[j];
      }

    /*
     * FIX 1 — reparametrisation gradients.
     * d_z already carries the 1/bsz factor from the output layer above.
     * Only the KL terms are multiplied by scale to batch-normalise them.
     * The original code mistakenly applied scale to the entire expression,
     * dividing the reconstruction path by bsz a second time.
     */
    float d_mu[LATENT], d_lv[LATENT];
    for (int i = 0; i < LATENT; i++) {
      float sig = expf(0.5f * lv_s[i]);
      d_mu[i] = d_z[i] + (beta * mu_s[i] / LATENT) * scale;
      d_lv[i] = d_z[i] * 0.5f * sig * eps_s[i] +
                (beta * 0.5f * (expf(lv_s[i]) - 1.0f) / LATENT) * scale;
      m->d_mub[i] += d_mu[i];
      m->d_lvb[i] += d_lv[i];
    }

    /* --- mu and lv heads --- */
    float d_eh2[H2];
    memset(d_eh2, 0, H2 * sizeof(float));
    for (int i = 0; i < H2; i++)
      for (int j = 0; j < LATENT; j++) {
        m->d_muw[i * LATENT + j] += enc_h2s[i] * d_mu[j];
        m->d_lvw[i * LATENT + j] += enc_h2s[i] * d_lv[j];
        d_eh2[i] += m->mu_w[i * LATENT + j] * d_mu[j] +
                    m->lv_w[i * LATENT + j] * d_lv[j];
      }

    /* --- encoder layer 2 --- */
    float d_peh2[H2];
    for (int i = 0; i < H2; i++)
      d_peh2[i] = d_eh2[i] * elu_d(peh2_s[i]);
    float d_eh1[H1];
    memset(d_eh1, 0, H1 * sizeof(float));
    for (int j = 0; j < H2; j++)
      m->d_eb2[j] += d_peh2[j];
    for (int i = 0; i < H1; i++)
      for (int j = 0; j < H2; j++) {
        m->d_ew2[i * H2 + j] += enc_h1s[i] * d_peh2[j];
        d_eh1[i] += m->enc_w2[i * H2 + j] * d_peh2[j];
      }

    /* --- encoder layer 1 --- */
    float d_peh1[H1];
    for (int i = 0; i < H1; i++)
      d_peh1[i] = d_eh1[i] * elu_d(peh1_s[i]);
    for (int j = 0; j < H1; j++)
      m->d_eb1[j] += d_peh1[j];
    for (int i = 0; i < ENC_IN; i++)
      for (int j = 0; j < H1; j++)
        m->d_ew1[i * H1 + j] += enc_in_s[i] * d_peh1[j];
  }
}

/* ------------------------------------------------------------------ */
/* Apply gradients                                                      */
/* ------------------------------------------------------------------ */
void apply_gradients(VAE *m, float lr) {
  m->adam_t++;
  int t = m->adam_t;
  adam_update(m->dec_w3, m->dw3, m->m_dw3, m->v_dw3, H2 * IMAGE_SIZE, lr, t);
  adam_update(m->dec_b3, m->db3, m->m_db3, m->v_db3, IMAGE_SIZE, lr, t);
  adam_update(m->dec_w2, m->dw2, m->m_dw2, m->v_dw2, H1 * H2, lr, t);
  adam_update(m->dec_b2, m->db2, m->m_db2, m->v_db2, H2, lr, t);
  adam_update(m->dec_w1, m->dw1, m->m_dw1, m->v_dw1, DEC_IN * H1, lr, t);
  adam_update(m->dec_b1, m->db1, m->m_db1, m->v_db1, H1, lr, t);
  adam_update(m->mu_w, m->d_muw, m->m_muw, m->v_muw, H2 * LATENT, lr, t);
  adam_update(m->mu_b, m->d_mub, m->m_mub, m->v_mub, LATENT, lr, t);
  adam_update(m->lv_w, m->d_lvw, m->m_lvw, m->v_lvw, H2 * LATENT, lr, t);
  adam_update(m->lv_b, m->d_lvb, m->m_lvb, m->v_lvb, LATENT, lr, t);
  adam_update(m->enc_w2, m->d_ew2, m->m_ew2, m->v_ew2, H1 * H2, lr, t);
  adam_update(m->enc_b2, m->d_eb2, m->m_eb2, m->v_eb2, H2, lr, t);
  adam_update(m->enc_w1, m->d_ew1, m->m_ew1, m->v_ew1, ENC_IN * H1, lr, t);
  adam_update(m->enc_b1, m->d_eb1, m->m_eb1, m->v_eb1, H1, lr, t);
}

/* ------------------------------------------------------------------ */
/* Model save / load                                                    */
/*                                                                      */
/* Checkpoint format (binary, little-endian):                           */
/*   [uint32] magic   = 0x45415643  ('CVAE')                           */
/*   [uint32] version = 2                                               */
/*   [float*] weights  (all layers, in encoder->latent->decoder order) */
/*   [uint32] adam_t                                                    */
/*   [float*] Adam m_ and v_ moment buffers (same layer order)         */
/* ------------------------------------------------------------------ */

#define CKPT_MAGIC 0x45415643u /* 'C','V','A','E' */
#define CKPT_VERSION 2u

void save_model(VAE *m, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "[WARN] cannot open %s for writing\n", path);
    return;
  }

/* write N floats; bail on short write */
#define WA(arr, n)                                                             \
  if (fwrite(arr, sizeof(float), (size_t)(n), f) != (size_t)(n)) {             \
    fprintf(stderr, "[ERROR] save failed at %s:%d\n", path, __LINE__);         \
    fclose(f);                                                                 \
    return;                                                                    \
  }
/* write one uint32_t */
#define WU(val)                                                                \
  {                                                                            \
    uint32_t _u = (uint32_t)(val);                                             \
    if (fwrite(&_u, sizeof _u, 1, f) != 1) {                                   \
      fprintf(stderr, "[ERROR] save failed at %s:%d\n", path, __LINE__);       \
      fclose(f);                                                               \
      return;                                                                  \
    }                                                                          \
  }

  WU(CKPT_MAGIC);
  WU(CKPT_VERSION);

  /* --- weights --- */
  WA(m->enc_w1, ENC_IN * H1);
  WA(m->enc_b1, H1);
  WA(m->enc_w2, H1 * H2);
  WA(m->enc_b2, H2);
  WA(m->mu_w, H2 * LATENT);
  WA(m->mu_b, LATENT);
  WA(m->lv_w, H2 * LATENT);
  WA(m->lv_b, LATENT);
  WA(m->dec_w1, DEC_IN * H1);
  WA(m->dec_b1, H1);
  WA(m->dec_w2, H1 * H2);
  WA(m->dec_b2, H2);
  WA(m->dec_w3, H2 * IMAGE_SIZE);
  WA(m->dec_b3, IMAGE_SIZE);

  /* --- Adam optimizer state --- */
  WU(m->adam_t);
  WA(m->m_ew1, ENC_IN * H1);
  WA(m->v_ew1, ENC_IN * H1);
  WA(m->m_eb1, H1);
  WA(m->v_eb1, H1);
  WA(m->m_ew2, H1 * H2);
  WA(m->v_ew2, H1 * H2);
  WA(m->m_eb2, H2);
  WA(m->v_eb2, H2);
  WA(m->m_muw, H2 * LATENT);
  WA(m->v_muw, H2 * LATENT);
  WA(m->m_mub, LATENT);
  WA(m->v_mub, LATENT);
  WA(m->m_lvw, H2 * LATENT);
  WA(m->v_lvw, H2 * LATENT);
  WA(m->m_lvb, LATENT);
  WA(m->v_lvb, LATENT);
  WA(m->m_dw1, DEC_IN * H1);
  WA(m->v_dw1, DEC_IN * H1);
  WA(m->m_db1, H1);
  WA(m->v_db1, H1);
  WA(m->m_dw2, H1 * H2);
  WA(m->v_dw2, H1 * H2);
  WA(m->m_db2, H2);
  WA(m->v_db2, H2);
  WA(m->m_dw3, H2 * IMAGE_SIZE);
  WA(m->v_dw3, H2 * IMAGE_SIZE);
  WA(m->m_db3, IMAGE_SIZE);
  WA(m->v_db3, IMAGE_SIZE);

#undef WA
#undef WU
  fclose(f);
  printf("[INFO] checkpoint saved -> %s  (adam_t=%d)\n", path, m->adam_t);
}

int load_model(VAE *m, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return 0;

/* read N floats; bail on short read */
#define RA(arr, n)                                                             \
  if (fread(arr, sizeof(float), (size_t)(n), f) != (size_t)(n)) {              \
    fprintf(stderr, "[WARN] checkpoint read error: %s\n", path);               \
    fclose(f);                                                                 \
    return 0;                                                                  \
  }
/* read one uint32_t into an int variable */
#define RU(var)                                                                \
  {                                                                            \
    uint32_t _u;                                                               \
    if (fread(&_u, sizeof _u, 1, f) != 1) {                                    \
      fprintf(stderr, "[WARN] checkpoint read error: %s\n", path);             \
      fclose(f);                                                               \
      return 0;                                                                \
    }                                                                          \
    (var) = (int)_u;                                                           \
  }

  int magic, version;
  RU(magic);
  RU(version);
  if ((uint32_t)magic != CKPT_MAGIC) {
    fprintf(stderr, "[WARN] not a CVAE checkpoint (bad magic=0x%08X): %s\n",
            (unsigned)magic, path);
    fclose(f);
    return 0;
  }
  if (version != (int)CKPT_VERSION) {
    fprintf(stderr, "[WARN] checkpoint version %d unsupported (want %u): %s\n",
            version, CKPT_VERSION, path);
    fclose(f);
    return 0;
  }

  /* --- weights --- */
  RA(m->enc_w1, ENC_IN * H1);
  RA(m->enc_b1, H1);
  RA(m->enc_w2, H1 * H2);
  RA(m->enc_b2, H2);
  RA(m->mu_w, H2 * LATENT);
  RA(m->mu_b, LATENT);
  RA(m->lv_w, H2 * LATENT);
  RA(m->lv_b, LATENT);
  RA(m->dec_w1, DEC_IN * H1);
  RA(m->dec_b1, H1);
  RA(m->dec_w2, H1 * H2);
  RA(m->dec_b2, H2);
  RA(m->dec_w3, H2 * IMAGE_SIZE);
  RA(m->dec_b3, IMAGE_SIZE);

  /* --- Adam optimizer state --- */
  int t;
  RU(t);
  m->adam_t = t;
  RA(m->m_ew1, ENC_IN * H1);
  RA(m->v_ew1, ENC_IN * H1);
  RA(m->m_eb1, H1);
  RA(m->v_eb1, H1);
  RA(m->m_ew2, H1 * H2);
  RA(m->v_ew2, H1 * H2);
  RA(m->m_eb2, H2);
  RA(m->v_eb2, H2);
  RA(m->m_muw, H2 * LATENT);
  RA(m->v_muw, H2 * LATENT);
  RA(m->m_mub, LATENT);
  RA(m->v_mub, LATENT);
  RA(m->m_lvw, H2 * LATENT);
  RA(m->v_lvw, H2 * LATENT);
  RA(m->m_lvb, LATENT);
  RA(m->v_lvb, LATENT);
  RA(m->m_dw1, DEC_IN * H1);
  RA(m->v_dw1, DEC_IN * H1);
  RA(m->m_db1, H1);
  RA(m->v_db1, H1);
  RA(m->m_dw2, H1 * H2);
  RA(m->v_dw2, H1 * H2);
  RA(m->m_db2, H2);
  RA(m->v_db2, H2);
  RA(m->m_dw3, H2 * IMAGE_SIZE);
  RA(m->v_dw3, H2 * IMAGE_SIZE);
  RA(m->m_db3, IMAGE_SIZE);
  RA(m->v_db3, IMAGE_SIZE);

#undef RA
#undef RU
  fclose(f);
  printf("[INFO] checkpoint loaded <- %s  (resuming from adam_t=%d)\n", path,
         m->adam_t);
  return 1;
}

/* ------------------------------------------------------------------ */
/* Generation                                                           */
/* ------------------------------------------------------------------ */

/*
 * Generate one image conditioned on `label`, sample from N(0, I).
 * temp: sampling temperature (1.0 = standard, lower = less varied).
 */
void generate_digit(VAE *m, int label, float temp, int epoch, int sample_idx) {
  float cond[NUM_CLASSES];
  one_hot(cond, label);

  float z[LATENT];
  for (int i = 0; i < LATENT; i++)
    z[i] = randn() * temp;

  /* Use slab buffers (row 0, bsz=1) for decoder-only pass */
  memcpy(m->dec_in_buf, z, LATENT * sizeof(float));
  memcpy(m->dec_in_buf + LATENT, cond, NUM_CLASSES * sizeof(float));

  linear(m->pre_dh1, m->dec_in_buf, m->dec_w1, m->dec_b1, DEC_IN, H1);
  for (int i = 0; i < H1; i++)
    m->dec_h1[i] = elu(m->pre_dh1[i]);
  linear(m->pre_dh2, m->dec_h1, m->dec_w2, m->dec_b2, H1, H2);
  for (int i = 0; i < H2; i++)
    m->dec_h2[i] = elu(m->pre_dh2[i]);
  linear(m->pre_out, m->dec_h2, m->dec_w3, m->dec_b3, H2, IMAGE_SIZE);
  for (int i = 0; i < IMAGE_SIZE; i++)
    m->output[i] = sigmoid(m->pre_out[i]);

  char path[256];
  snprintf(path, sizeof path, "%s/epoch_%03d_digit%d_s%d.pgm", RESULT_DIR,
           epoch, label, sample_idx);
  FILE *f = fopen(path, "w");
  if (!f)
    return;
  fprintf(f, "P2\n28 28\n255\n");
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++)
      fprintf(f, "%d ", (int)(m->output[y * 28 + x] * 255));
    fprintf(f, "\n");
  }
  fclose(f);
}

/* Save 3 samples per digit class every SAVE_EVERY epochs */
void generate_epoch_samples(VAE *m, int epoch) {
  for (int cls = 0; cls < NUM_CLASSES; cls++)
    for (int s = 0; s < 3; s++)
      generate_digit(m, cls, 0.8f, epoch, s);
  printf("[EPOCH %3d] samples -> %s/epoch_%03d_digit*_s*.pgm\n", epoch,
         RESULT_DIR, epoch);
}

/* ------------------------------------------------------------------ */
/* Dataset                                                              */
/* ------------------------------------------------------------------ */
#include "mnist_loader.h"

Dataset *load_dataset(void) {
  Dataset *ds = malloc(sizeof(Dataset));
#ifdef FULL_MNIST
  printf("[INFO] loading full MNIST (0-9)...\n");
#else
  printf("[INFO] loading binary MNIST (0-1)...\n");
#endif
  if (load_mnist_data("data/train-images-idx3-ubyte",
                      "data/train-labels-idx1-ubyte", &ds->images, &ds->labels,
                      &ds->count)) {
    printf("[INFO] loaded %d images\n", ds->count);
    return ds;
  }
  fprintf(stderr, "[ERROR] MNIST data not found — run ./download_mnist.sh\n");
  free(ds);
  return NULL;
}

void free_dataset(Dataset *ds) {
  if (!ds)
    return;
  for (int i = 0; i < ds->count; i++)
    free(ds->images[i]);
  free(ds->images);
  free(ds->labels);
  free(ds);
}

/* ------------------------------------------------------------------ */
/* Filesystem helpers                                                   */
/* ------------------------------------------------------------------ */

/* Portable recursive mkdir — no shell, no system() */
static void mkdir_p(const char *path) {
  char tmp[256];
  snprintf(tmp, sizeof tmp, "%s", path);
  for (char *p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0';
      mkdir(tmp, 0755);
      *p = '/';
    }
  }
  mkdir(tmp, 0755);
}

/* ------------------------------------------------------------------ */
/* Training loop                                                        */
/* ------------------------------------------------------------------ */
void train(VAE *m, Dataset *ds) {
  mkdir_p(MODEL_DIR);
  mkdir_p(RESULT_DIR);

  /* ---- validation split ------------------------------------------ */
  /* One-time shuffle of the full dataset so the val set is random.    */
  /* Val = last 10%; train = first 90%.  Only the train portion is     */
  /* reshuffled each epoch, keeping val samples stable throughout.      */
  for (int i = ds->count - 1; i > 0; i--) {
    int j = randi(i + 1);
    float *ti = ds->images[i];
    ds->images[i] = ds->images[j];
    ds->images[j] = ti;
    int tl = ds->labels[i];
    ds->labels[i] = ds->labels[j];
    ds->labels[j] = tl;
  }
  int val_count = ds->count / 10;
  int train_count = ds->count - val_count;
  printf(
      "[TRAIN] %d train  %d val  |  %d epochs  lr=%.4f  batch=%d  classes=%d\n",
      train_count, val_count, EPOCHS, (double)LR, BATCH_SIZE, NUM_CLASSES);

  float best_val_loss = 1e9f;
  int patience = 0;

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    clock_t t0 = clock();

    /* LR: linear warmup (0..30) then cosine decay, floor at 5% */
    float lr;
    if (epoch < 30) {
      lr = LR * (epoch + 1) / 30.0f;
    } else {
      float p = (float)(epoch - 30) / (float)(EPOCHS - 30);
      lr = LR * 0.5f * (1.0f + cosf(3.14159265f * p));
      if (lr < LR * 0.05f)
        lr = LR * 0.05f;
    }

    /* beta: KL annealing from BETA_START to BETA_END over BETA_ANNEAL epochs */
    float beta = BETA_START;
    if (epoch > BETA_WARMUP) {
      float p = (float)(epoch - BETA_WARMUP) / (float)BETA_ANNEAL;
      if (p > 1.0f)
        p = 1.0f;
      beta = BETA_START + (BETA_END - BETA_START) * p;
    }

    /* shuffle training portion only (Fisher-Yates) */
    for (int i = train_count - 1; i > 0; i--) {
      int j = randi(i + 1);
      float *ti = ds->images[i];
      ds->images[i] = ds->images[j];
      ds->images[j] = ti;
      int tl = ds->labels[i];
      ds->labels[i] = ds->labels[j];
      ds->labels[j] = tl;
    }

    /* ---- forward + backward on training set ---- */
    float total_loss = 0.0f;
    int nb = (train_count + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int b = 0; b < nb; b++) {
      int start = b * BATCH_SIZE;
      int end = start + BATCH_SIZE;
      if (end > train_count)
        end = train_count;
      int bsz = end - start;

      /* True batched forward: all bsz samples processed layer-by-layer
       * in a single linear_batch() call each, not one at a time. */
      reset_grads(m);
      vae_forward(m, &ds->images[start], &ds->labels[start], bsz, 1);
      float bloss = vae_loss(m, &ds->images[start], bsz, beta);
      if (isfinite(bloss))
        vae_backward(m, &ds->images[start], &ds->labels[start], bsz, beta);
      else
        bloss = 1.0f;
      total_loss += bloss;
      apply_gradients(m, lr);
    }
    float train_avg = total_loss / nb;

    /* ---- validation loss (forward only, no noise, no backprop) ---- */
    float val_loss = 0.0f;
    int val_ok = 0;
    for (int i = train_count; i < ds->count; i++) {
      /* single-sample batch-of-1 forward pass */
      vae_forward(m, &ds->images[i], &ds->labels[i], 1, 0);
      float l = vae_loss(m, &ds->images[i], 1, beta);
      if (isfinite(l)) {
        val_loss += l;
        val_ok++;
      }
    }
    val_loss = val_ok > 0 ? val_loss / val_ok : 1e9f;

    /* ---- early stopping on val loss (not train loss) ---- */
    if (val_loss < best_val_loss) {
      best_val_loss = val_loss;
      patience = 0;
      save_model(m, MODEL_FILE);
    } else {
      patience++;
      if (patience > 60 && epoch > 150) {
        printf("[TRAIN] early stop epoch %d  best_val=%.4f\n", epoch,
               (double)best_val_loss);
        break;
      }
    }

    double sps = train_count / ((double)(clock() - t0) / CLOCKS_PER_SEC);
    if (epoch % 10 == 0 || epoch < 10)
      printf("[EPOCH %3d] train=%.4f  val=%.4f  best=%.4f"
             "  beta=%.5f  lr=%.5f  %.0f img/s\n",
             epoch, (double)train_avg, (double)val_loss, (double)best_val_loss,
             (double)beta, (double)lr, sps);

    if (epoch % SAVE_EVERY == 0)
      generate_epoch_samples(m, epoch);
  }
  printf("[TRAIN] done  best_val=%.4f  model=%s\n", (double)best_val_loss,
         MODEL_FILE);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
  srand((unsigned)time(NULL));
  rng = (uint64_t)time(NULL) ^ 0xDEADBEEFULL;

  printf(
      "CVAE %s  enc=[%d]->%d->%d->[%dz]  dec=[%dz]->%d->%d->[%d]  classes=%d\n",
      VERSION, ENC_IN, H1, H2, LATENT, DEC_IN, H1, H2, IMAGE_SIZE, NUM_CLASSES);

  VAE *m = create_vae();
  load_model(m, MODEL_FILE); /* resume if exists */

  Dataset *ds = load_dataset();
  if (!ds) {
    free_vae(m);
    return 1;
  }

  train(m, ds);

  /* final generation: 3 samples per digit */
  printf("\n[GEN] final samples per digit...\n");
  for (int cls = 0; cls < NUM_CLASSES; cls++)
    for (int s = 0; s < 3; s++)
      generate_digit(m, cls, 0.8f, 9999, s);
  printf("[GEN] done -> %s/epoch_9999_digit*_s*.pgm\n", RESULT_DIR);

  free_dataset(ds);
  free_vae(m);
  return 0;
}