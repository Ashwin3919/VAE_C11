CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99
LDFLAGS = -lm
TARGET = diffusion_model
TARGET_V2 = diffusion_model_v2
TARGET_V3 = diffusion_model_v3
VIEWER = view_images
DATA_VIEWER = view_training_data
SOURCES = diffusion_model.c
SOURCES_V2 = diffusion_model_v2.c
SOURCES_V3 = diffusion_model_v3.c
VIEWER_SOURCES = view_images.c
DATA_VIEWER_SOURCES = view_training_data.c

all: $(TARGET) $(TARGET_V2) $(TARGET_V3) $(VIEWER) $(DATA_VIEWER)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

$(TARGET_V2): $(SOURCES_V2)
	$(CC) $(CFLAGS) -o $(TARGET_V2) $(SOURCES_V2) $(LDFLAGS)

$(TARGET_V3): $(SOURCES_V3)
	$(CC) $(CFLAGS) -o $(TARGET_V3) $(SOURCES_V3) $(LDFLAGS)

$(VIEWER): $(VIEWER_SOURCES)
	$(CC) $(CFLAGS) -o $(VIEWER) $(VIEWER_SOURCES)

$(DATA_VIEWER): $(DATA_VIEWER_SOURCES)
	$(CC) $(CFLAGS) -o $(DATA_VIEWER) $(DATA_VIEWER_SOURCES)

clean:
	rm -f $(TARGET) $(TARGET_V2) $(TARGET_V3) $(VIEWER) $(DATA_VIEWER) generated_*.pgm improved_*.pgm training_*.pgm weights_*.bin

run: $(TARGET)
	./$(TARGET)

run-v2: $(TARGET_V2)
	./$(TARGET_V2)

run-v3: $(TARGET_V3)
	./$(TARGET_V3)

view: $(VIEWER)
	./$(VIEWER)

view-data: $(DATA_VIEWER)
	./$(DATA_VIEWER)

test: all
	./$(TARGET) && ./$(VIEWER)

test-v2: all
	./$(TARGET_V2) && ./$(VIEWER) improved_*.pgm

test-v3: all
	./$(TARGET_V3) && ./$(VIEWER) realistic_*.pgm

.PHONY: all clean run run-v2 run-v3 view view-data test test-v2 test-v3 