/* Optional verifier for the capture written by large_offline_workbench. */
#include <ghostty/vt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum Checkpoint {
    WELCOME, FAILURE, AXES_MISMATCH, SOURCE, RETURNED, GROUPS, GROUP_FLOW, ZOOM,
    FIRST_COMMIT, SECOND_PREVIEW, SECOND_COMMIT, INVALID_PORT, REJECTED, AXES_UNKNOWN,
    BEFORE_RECHECK, AFTER_RECHECK, RECHECK_EVIDENCE, RETURN_TO_COMPARISON, RETURN_TO_RECEIPT,
    AXES_EQUAL, CHECKPOINT_COUNT,
};

typedef struct {
    GhosttyKittyGraphicsPlacementRenderInfo geometry;
    GhosttyString pixels;
} GraphImage;

typedef struct {
    uint8_t *text;
    size_t length;
    GraphImage image;
} ReferenceFrame;

static void require(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "%s\n", message);
        exit(1);
    }
}

static size_t read_placements(GhosttyTerminal terminal, GraphImage *graph) {
    GhosttyKittyGraphics graphics;
    require(ghostty_terminal_get(terminal, GHOSTTY_TERMINAL_DATA_KITTY_GRAPHICS,
                                 &graphics) == GHOSTTY_SUCCESS,
            "Kitty graphics storage is unavailable");
    GhosttyKittyGraphicsPlacementIterator iterator;
    require(ghostty_kitty_graphics_placement_iterator_new(NULL, &iterator) == GHOSTTY_SUCCESS,
            "Cannot create placement iterator");
    require(ghostty_kitty_graphics_get(graphics, GHOSTTY_KITTY_GRAPHICS_DATA_PLACEMENT_ITERATOR,
                                      &iterator) == GHOSTTY_SUCCESS,
            "Cannot read Kitty placements");
    size_t count = 0;
    while (ghostty_kitty_graphics_placement_next(iterator)) {
        count++;
        if (!graph) continue;
        uint32_t id;
        require(ghostty_kitty_graphics_placement_get(iterator,
                    GHOSTTY_KITTY_GRAPHICS_PLACEMENT_DATA_IMAGE_ID, &id) == GHOSTTY_SUCCESS,
                "Cannot read graph image ID");
        GhosttyKittyGraphicsImage image = ghostty_kitty_graphics_image(graphics, id);
        require(image && ghostty_kitty_graphics_image_get(image,
                    GHOSTTY_KITTY_GRAPHICS_IMAGE_DATA_PIXELS, &graph->pixels) == GHOSTTY_SUCCESS,
                "Cannot read decoded graph pixels");
        require(graph->pixels.len > 0, "Empty graph image");
        graph->geometry = (GhosttyKittyGraphicsPlacementRenderInfo)
            GHOSTTY_INIT_SIZED(GhosttyKittyGraphicsPlacementRenderInfo);
        require(ghostty_kitty_graphics_placement_render_info(iterator, image, terminal,
                    &graph->geometry) == GHOSTTY_SUCCESS && graph->geometry.viewport_visible,
                "Graph image is outside the viewport");
    }
    ghostty_kitty_graphics_placement_iterator_free(iterator);
    return count;
}

static bool contains(const uint8_t *text, size_t size, const char *expected) {
    const size_t length = strlen(expected);
    for (size_t i = 0; length <= size && i <= size - length; i++) {
        if (memcmp(text + i, expected, length) == 0) {
            return true;
        }
    }
    return false;
}

static size_t graph_text_length(const uint8_t *text, size_t length) {
    size_t rows = 0;
    for (size_t i = 0; i < length; i++) {
        if (text[i] == '\n' && ++rows == 23) return i + 1;
    }
    require(false, "Missing workbench graph rows");
    return 0;
}

static uint8_t *copy_bytes(const uint8_t *data, size_t length) {
    uint8_t *copy = malloc(length);
    require(copy != NULL, "Cannot save reference frame");
    memcpy(copy, data, length);
    return copy;
}

static void check_graph(ReferenceFrame *reference, const uint8_t *text, size_t length,
                        GraphImage image, bool save, bool pixels) {
    if (save) {
        reference->text = copy_bytes(text, length);
        reference->length = length;
        reference->image = image;
        if (pixels) reference->image.pixels.ptr = copy_bytes(image.pixels.ptr, image.pixels.len);
        return;
    }
    require(reference->length == length && memcmp(reference->text, text, length) == 0,
            "Restored terminal cells differ from the reference");
    if (pixels) {
        GhosttyKittyGraphicsPlacementRenderInfo before = reference->image.geometry;
        GhosttyKittyGraphicsPlacementRenderInfo after = image.geometry;
        require(before.viewport_col == after.viewport_col && before.viewport_row == after.viewport_row &&
                    before.pixel_width == after.pixel_width && before.pixel_height == after.pixel_height,
                "Restored pixel graph moved or changed size");
        require(reference->image.pixels.len == image.pixels.len &&
                    memcmp(reference->image.pixels.ptr, image.pixels.ptr, image.pixels.len) == 0,
                "Restored graph pixels differ from the reference");
    }
    free(reference->text);
    free((void *)reference->image.pixels.ptr);
    *reference = (ReferenceFrame){0};
}

static void show_checkpoint(GhosttyTerminal terminal, const char *name, size_t step, bool pixels,
                            ReferenceFrame *reference) {
    static const char *evidence[] = {
        "Open workbench", "model.head #2", "2×48 (observed) ↔ …×32 (declared)",
        "model.encoder[0]", "1–3/3", "/34", "/32", "110%", "v2 · repair-1",
        "net.merge.in[0]", "v3 · repair-2", "net.adapter.out[9]", "source port unavailable · v3", "? ↔ …×48 (declared)",
        "model.encoder.out[0]", "net.adapter.out[0]", "Target net.merge.in[0]",
        "› After · ✓ Sample passed", "Passed · 2×48", "2×48 (observed) ↔ …×48 (declared)",
    };
    require(step < CHECKPOINT_COUNT, "Unexpected checkpoint");
    GhosttyFormatter formatter;
    require(ghostty_formatter_terminal_new(NULL, &formatter, terminal) == GHOSTTY_SUCCESS,
            "Cannot read terminal cells");
    const GhosttyFormatterFormat format = GHOSTTY_FORMATTER_FORMAT_PLAIN;
    const bool trim = true;
    require(ghostty_formatter_set(formatter, GHOSTTY_FORMATTER_OPT_EMIT, &format) == GHOSTTY_SUCCESS,
            "Cannot select plain text");
    require(ghostty_formatter_set(formatter, GHOSTTY_FORMATTER_OPT_TRIM, &trim) == GHOSTTY_SUCCESS,
            "Cannot trim blank cells");
    uint8_t *text = NULL;
    size_t length = 0;
    require(ghostty_formatter_format_alloc(formatter, NULL, &text, &length) == GHOSTTY_SUCCESS,
            "Cannot format terminal cells");
    const bool save = step == INVALID_PORT || step == AFTER_RECHECK;
    const bool compare = step == REJECTED || step == RETURN_TO_COMPARISON;
    GraphImage image = {0};
    const size_t placements = read_placements(terminal, pixels && (save || compare) ? &image : NULL);
    printf("=== %s placements=%zu ===\n", name, placements);
    fwrite(text, 1, length, stdout);
    putchar('\n');
    require(contains(text, length, evidence[step]), evidence[step]);
    const bool graph_visible = step != WELCOME && step != RECHECK_EVIDENCE;
    require(placements == (size_t)(pixels && graph_visible), "Unexpected image placement count");
    if (step == ZOOM) {
        require(contains(text, length, "generated tensor flow v1"), "Zoom lost the active group view");
        require(contains(text, length, "model.head #2"), "Zoom lost the retained call details");
    }
    if (step == FIRST_COMMIT || step == SECOND_PREVIEW) {
        require(contains(text, length, "tensor flow v2"), "First committed graph is missing");
    }
    if (step == SECOND_COMMIT || step == INVALID_PORT || step == REJECTED) {
        require(contains(text, length, "tensor flow v3"), "Second committed graph is missing");
        require(!contains(text, length, "tensor flow v4"), "Unexpected revision after second commit");
    }
    if (step == FIRST_COMMIT) {
        require(contains(text, length, "net.head.in[0] · …×32 → …×48"), "First receipt is missing the edited port");
    }
    if (step == SOURCE || step == RETURNED) {
        require(contains(text, length, "model.encoder[0] → net.head[0]"), "Trace selected the wrong wire");
        require(contains(text, length, "model.head #2"), "Trace changed the call identity");
    }
    if (step == RECHECK_EVIDENCE) {
        require(contains(text, length, "Recheck evidence"), "Recheck transcript is missing");
        require(contains(text, length, "model.encoder.out[0] → net.merge.in[0]"), "Original producer is missing");
        require(contains(text, length, "net.adapter.out[0] → net.merge.in[0]"), "Rewired producer is missing");
        require(!contains(text, length, "tensor flow v"), "Network scene was left behind");
    }
    if (step == RETURN_TO_COMPARISON || step == RETURN_TO_RECEIPT) {
        require(contains(text, length, "observed tensor flow v3"), "Committed graph was not restored");
        require(!contains(text, length, "Recheck evidence"), "Transcript was left behind");
    }
    if (step == RETURN_TO_RECEIPT) {
        require(!contains(text, length, "Target observed"), "Comparison was left behind");
    }
    if (step == AXES_MISMATCH || step == AXES_UNKNOWN || step == AXES_EQUAL) {
        require(contains(text, length, "Tensor cross-sections"), "Axis scene is missing");
        require(!contains(text, length, "tensor flow v"), "Network scene was left behind");
    }
    if (save || compare) {
        size_t compare_length = (step == INVALID_PORT || step == REJECTED)
            ? graph_text_length(text, length) : length;
        check_graph(reference, text, compare_length, image, save, pixels);
    }
    ghostty_free(NULL, text, length);
    ghostty_formatter_free(formatter);
}

int main(int argc, char **argv) {
    require(argc == 4, "Usage: ghostty-replay stream.ansi checkpoints.txt cells|pixels");
    const bool pixels = strcmp(argv[3], "pixels") == 0;
    require(pixels || strcmp(argv[3], "cells") == 0, "Unknown rendering mode");
    FILE *stream = fopen(argv[1], "rb");
    FILE *index = fopen(argv[2], "r");
    require(stream && index, "Cannot open replay capture");
    GhosttyTerminal terminal;
    require(ghostty_terminal_new(NULL, &terminal, 100, 30) == GHOSTTY_SUCCESS,
            "Cannot create test terminal");
    require(ghostty_terminal_resize(terminal, 100, 30, 9, 18) == GHOSTTY_SUCCESS,
            "Cannot set test pixel geometry");

    size_t offset = 0;
    size_t end;
    size_t step = 0;
    ReferenceFrame reference = {0};
    char name[96];
    uint8_t buffer[8192];
    while (fscanf(index, "%zu %95s", &end, name) == 2) {
        require(end > offset, "Checkpoint offsets must increase");
        while (offset < end) {
            size_t size = end - offset;
            if (size > sizeof(buffer)) {
                size = sizeof(buffer);
            }
            require(fread(buffer, 1, size, stream) == size, "Incomplete replay stream");
            ghostty_terminal_vt_write(terminal, buffer, size);
            offset += size;
        }
        show_checkpoint(terminal, name, step, pixels, &reference);
        step++;
    }
    require(feof(index) && step == CHECKPOINT_COUNT, "Invalid checkpoint list");
    ghostty_terminal_free(terminal);
    fclose(index);
    fclose(stream);
    return 0;
}
