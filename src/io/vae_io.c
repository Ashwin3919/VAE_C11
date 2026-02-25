/*
 * vae_io.c — checkpoint save / load + mkdir_p.
 *
 * Checkpoint format (binary, explicit little-endian):
 *   [uint32 LE] magic   = 0x45415643  ('C','V','A','E')
 *   [uint32 LE] version = 4
 *   [float* LE] weights  (encoder → latent heads → decoder, same order as slab)
 *   [uint32 LE] adam_t
 *   [float* LE] Adam m_ and v_ moment buffers (same layer order)
 *
 * All multi-byte values are stored in canonical little-endian byte order so
 * that checkpoints are portable across architectures (x86, ARM, big-endian
 * PowerPC, etc.).  The host byte order is irrelevant at read or write time.
 */
#include "vae_io.h"
#include "vae_math.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define CKPT_MAGIC   0x45415643u /* 'CVAE' */
#define CKPT_VERSION 4u          /* v4: explicit LE encoding */

/* ── endian-safe I/O primitives ─────────────────────────────────────── */

static int write_le_u32(uint32_t v, FILE *f) {
  uint8_t buf[4] = {
    (uint8_t)(v),
    (uint8_t)(v >> 8),
    (uint8_t)(v >> 16),
    (uint8_t)(v >> 24)
  };
  return fwrite(buf, 1, 4, f) == 4;
}

static int read_le_u32(uint32_t *out, FILE *f) {
  uint8_t buf[4];
  if (fread(buf, 1, 4, f) != 4)
    return 0;
  *out = (uint32_t)buf[0]
       | ((uint32_t)buf[1] << 8)
       | ((uint32_t)buf[2] << 16)
       | ((uint32_t)buf[3] << 24);
  return 1;
}

/* Buffered float write: encode each float as 4 LE bytes.
 * Uses a 1 KB stack buffer to keep syscall count low. */
static int write_le_floats(const float *arr, int n, FILE *f) {
  enum { CHUNK = 256 }; /* floats per batch */
  uint8_t buf[CHUNK * 4];
  for (int base = 0; base < n; base += CHUNK) {
    int cnt = (base + CHUNK <= n) ? CHUNK : (n - base);
    for (int i = 0; i < cnt; i++) {
      uint32_t bits;
      memcpy(&bits, &arr[base + i], 4);
      buf[i * 4 + 0] = (uint8_t)(bits);
      buf[i * 4 + 1] = (uint8_t)(bits >> 8);
      buf[i * 4 + 2] = (uint8_t)(bits >> 16);
      buf[i * 4 + 3] = (uint8_t)(bits >> 24);
    }
    if (fwrite(buf, 1, (size_t)cnt * 4, f) != (size_t)cnt * 4)
      return 0;
  }
  return 1;
}

static int read_le_floats(float *arr, int n, FILE *f) {
  enum { CHUNK = 256 };
  uint8_t buf[CHUNK * 4];
  for (int base = 0; base < n; base += CHUNK) {
    int cnt = (base + CHUNK <= n) ? CHUNK : (n - base);
    if (fread(buf, 1, (size_t)cnt * 4, f) != (size_t)cnt * 4)
      return 0;
    for (int i = 0; i < cnt; i++) {
      uint32_t bits = (uint32_t)buf[i * 4 + 0]
                    | ((uint32_t)buf[i * 4 + 1] << 8)
                    | ((uint32_t)buf[i * 4 + 2] << 16)
                    | ((uint32_t)buf[i * 4 + 3] << 24);
      memcpy(&arr[base + i], &bits, 4);
    }
  }
  return 1;
}

/* ── write helpers ──────────────────────────────────────────────────── */
#define WA(arr, n)                                                             \
  if (!write_le_floats((arr), (n), f)) {                                       \
    fprintf(stderr, "[ERROR] save failed at %s:%d\n", path, __LINE__);         \
    fclose(f);                                                                 \
    return 0;                                                                  \
  }
#define WU(val)                                                                \
  if (!write_le_u32((uint32_t)(val), f)) {                                     \
    fprintf(stderr, "[ERROR] save failed at %s:%d\n", path, __LINE__);         \
    fclose(f);                                                                 \
    return 0;                                                                  \
  }

int save_model(const VAE *m, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "[WARN] cannot open %s for writing\n", path);
    return 0;
  }
  const VAEConfig *c = &m->cfg;

  WU(CKPT_MAGIC);
  WU(CKPT_VERSION);
  /* weights */
  WA(m->enc_w1, c->enc_in * c->h1);
  WA(m->enc_b1, c->h1);
  WA(m->enc_w2, c->h1 * c->h2);
  WA(m->enc_b2, c->h2);
  WA(m->mu_w, c->h2 * c->latent);
  WA(m->mu_b, c->latent);
  WA(m->lv_w, c->h2 * c->latent);
  WA(m->lv_b, c->latent);
  WA(m->dec_w1, c->dec_in * c->h1);
  WA(m->dec_b1, c->h1);
  WA(m->dec_w2, c->h1 * c->h2);
  WA(m->dec_b2, c->h2);
  WA(m->dec_w3, c->h2 * IMAGE_SIZE);
  WA(m->dec_b3, IMAGE_SIZE);
  /* Adam state */
  WU(m->adam_t);
  WA(m->m_ew1, c->enc_in * c->h1);
  WA(m->v_ew1, c->enc_in * c->h1);
  WA(m->m_eb1, c->h1);
  WA(m->v_eb1, c->h1);
  WA(m->m_ew2, c->h1 * c->h2);
  WA(m->v_ew2, c->h1 * c->h2);
  WA(m->m_eb2, c->h2);
  WA(m->v_eb2, c->h2);
  WA(m->m_muw, c->h2 * c->latent);
  WA(m->v_muw, c->h2 * c->latent);
  WA(m->m_mub, c->latent);
  WA(m->v_mub, c->latent);
  WA(m->m_lvw, c->h2 * c->latent);
  WA(m->v_lvw, c->h2 * c->latent);
  WA(m->m_lvb, c->latent);
  WA(m->v_lvb, c->latent);
  WA(m->m_dw1, c->dec_in * c->h1);
  WA(m->v_dw1, c->dec_in * c->h1);
  WA(m->m_db1, c->h1);
  WA(m->v_db1, c->h1);
  WA(m->m_dw2, c->h1 * c->h2);
  WA(m->v_dw2, c->h1 * c->h2);
  WA(m->m_db2, c->h2);
  WA(m->v_db2, c->h2);
  WA(m->m_dw3, c->h2 * IMAGE_SIZE);
  WA(m->v_dw3, c->h2 * IMAGE_SIZE);
  WA(m->m_db3, IMAGE_SIZE);
  WA(m->v_db3, IMAGE_SIZE);
  fclose(f);
  printf("[INFO] checkpoint saved -> %s  (adam_t=%d)\n", path, m->adam_t);
  return 1;
}
#undef WA
#undef WU

/* ── read helpers ──────────────────────────────────────────────────── */
#define RA(arr, n)                                                             \
  if (!read_le_floats((arr), (n), f)) {                                        \
    fprintf(stderr, "[WARN] checkpoint read error: %s\n", path);               \
    fclose(f);                                                                 \
    return 0;                                                                  \
  }
#define RU(var)                                                                \
  do {                                                                         \
    uint32_t _u;                                                               \
    if (!read_le_u32(&_u, f)) {                                                \
      fprintf(stderr, "[WARN] checkpoint read error: %s\n", path);             \
      fclose(f);                                                               \
      return 0;                                                                \
    }                                                                          \
    (var) = (int)_u;                                                           \
  } while (0)

int load_model(VAE *m, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return 0;
  const VAEConfig *c = &m->cfg;

  /* Read magic and version directly as uint32_t — no cast dance. */
  uint32_t magic, version;
  if (!read_le_u32(&magic, f) || !read_le_u32(&version, f)) {
    fprintf(stderr, "[WARN] checkpoint read error: %s\n", path);
    fclose(f);
    return 0;
  }
  if (magic != CKPT_MAGIC) {
    fprintf(stderr, "[WARN] not a CVAE checkpoint (bad magic=0x%08X): %s\n",
            magic, path);
    fclose(f);
    return 0;
  }
  if (version != CKPT_VERSION) {
    fprintf(stderr,
            "[WARN] checkpoint version %u unsupported (want %u): %s\n"
            "       Re-train to generate a v%u checkpoint.\n",
            version, CKPT_VERSION, path, CKPT_VERSION);
    fclose(f);
    return 0;
  }
  /* weights */
  RA(m->enc_w1, c->enc_in * c->h1);
  RA(m->enc_b1, c->h1);
  RA(m->enc_w2, c->h1 * c->h2);
  RA(m->enc_b2, c->h2);
  RA(m->mu_w, c->h2 * c->latent);
  RA(m->mu_b, c->latent);
  RA(m->lv_w, c->h2 * c->latent);
  RA(m->lv_b, c->latent);
  RA(m->dec_w1, c->dec_in * c->h1);
  RA(m->dec_b1, c->h1);
  RA(m->dec_w2, c->h1 * c->h2);
  RA(m->dec_b2, c->h2);
  RA(m->dec_w3, c->h2 * IMAGE_SIZE);
  RA(m->dec_b3, IMAGE_SIZE);
  /* Adam state */
  int t;
  RU(t);
  m->adam_t = t;
  RA(m->m_ew1, c->enc_in * c->h1);
  RA(m->v_ew1, c->enc_in * c->h1);
  RA(m->m_eb1, c->h1);
  RA(m->v_eb1, c->h1);
  RA(m->m_ew2, c->h1 * c->h2);
  RA(m->v_ew2, c->h1 * c->h2);
  RA(m->m_eb2, c->h2);
  RA(m->v_eb2, c->h2);
  RA(m->m_muw, c->h2 * c->latent);
  RA(m->v_muw, c->h2 * c->latent);
  RA(m->m_mub, c->latent);
  RA(m->v_mub, c->latent);
  RA(m->m_lvw, c->h2 * c->latent);
  RA(m->v_lvw, c->h2 * c->latent);
  RA(m->m_lvb, c->latent);
  RA(m->v_lvb, c->latent);
  RA(m->m_dw1, c->dec_in * c->h1);
  RA(m->v_dw1, c->dec_in * c->h1);
  RA(m->m_db1, c->h1);
  RA(m->v_db1, c->h1);
  RA(m->m_dw2, c->h1 * c->h2);
  RA(m->v_dw2, c->h1 * c->h2);
  RA(m->m_db2, c->h2);
  RA(m->v_db2, c->h2);
  RA(m->m_dw3, c->h2 * IMAGE_SIZE);
  RA(m->v_dw3, c->h2 * IMAGE_SIZE);
  RA(m->m_db3, IMAGE_SIZE);
  RA(m->v_db3, IMAGE_SIZE);
  fclose(f);
  printf("[INFO] checkpoint loaded <- %s  (resuming from adam_t=%d)\n", path,
         m->adam_t);
  return 1;
}
#undef RA
#undef RU

void mkdir_p(const char *path) {
  char tmp[PATH_BUF_SIZE];
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
