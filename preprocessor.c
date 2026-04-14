#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_PAGES        1000
#define MAX_LINE_LEN     1024
#define MAX_LINES_PER_PAGE 200
#define PAGE_DELIMITER   "---PAGE_BREAK---"

typedef struct {
    char lines[MAX_LINES_PER_PAGE][MAX_LINE_LEN];
    int  line_count;
} Page;

Page pages[MAX_PAGES];
int  page_count = 0;

typedef struct {
    char line[MAX_LINE_LEN];
    int  count;
} RepeatedLine;

RepeatedLine top_candidates[MAX_LINES_PER_PAGE * 10];
int          top_candidate_count = 0;
RepeatedLine bottom_candidates[MAX_LINES_PER_PAGE * 10];
int          bottom_candidate_count = 0;

/* In-place trim: removes leading and trailing whitespace */
void trim(char *str) {
    if (!str || *str == '\0') return;

    /* Trim leading whitespace */
    int start = 0;
    while (str[start] && isspace((unsigned char)str[start])) start++;

    /* Shift content left */
    if (start > 0) {
        int i = 0;
        while (str[start + i]) { str[i] = str[start + i]; i++; }
        str[i] = '\0';
    }

    /* Trim trailing whitespace */
    int end = (int)strlen(str) - 1;
    while (end >= 0 && isspace((unsigned char)str[end])) end--;
    str[end + 1] = '\0';
}

void add_candidate(RepeatedLine *candidates, int *count, const char *line) {
    for (int i = 0; i < *count; i++) {
        if (strcmp(candidates[i].line, line) == 0) {
            candidates[i].count++;
            return;
        }
    }
    if (*count < MAX_LINES_PER_PAGE * 10) {
        strncpy(candidates[*count].line, line, MAX_LINE_LEN - 1);
        candidates[*count].line[MAX_LINE_LEN - 1] = '\0';
        candidates[*count].count = 1;
        (*count)++;
    }
}

int is_header_footer(RepeatedLine *candidates, int count, const char *line, int total_pages) {
    if (total_pages < 2) return 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(candidates[i].line, line) == 0) {
            return (candidates[i].count >= 2 &&
                    (candidates[i].count > total_pages / 2 || total_pages <= 3));
        }
    }
    return 0;
}

int main(void) {
    char buffer[MAX_LINE_LEN];
    int  current_line_in_page = 0;

    /* ── Read input, split on PAGE_BREAK ── */
    while (fgets(buffer, sizeof(buffer), stdin)) {
        buffer[strcspn(buffer, "\n")] = '\0';

        if (strstr(buffer, PAGE_DELIMITER)) {
            if (page_count < MAX_PAGES) {
                page_count++;
                current_line_in_page = 0;
            }
            continue;
        }

        if (page_count < MAX_PAGES && current_line_in_page < MAX_LINES_PER_PAGE) {
            strncpy(pages[page_count].lines[current_line_in_page], buffer, MAX_LINE_LEN - 1);
            pages[page_count].lines[current_line_in_page][MAX_LINE_LEN - 1] = '\0';
            pages[page_count].line_count++;
            current_line_in_page++;
        }
    }
    /* flush last page if it never got a PAGE_BREAK */
    if (pages[page_count].line_count > 0) page_count++;

    /* ── Build header/footer candidate lists ── */
    for (int i = 0; i < page_count; i++) {
        for (int j = 0; j < 3 && j < pages[i].line_count; j++) {
            char trimmed[MAX_LINE_LEN];
            strncpy(trimmed, pages[i].lines[j], MAX_LINE_LEN - 1);
            trimmed[MAX_LINE_LEN - 1] = '\0';
            trim(trimmed);
            if (strlen(trimmed) > 0)
                add_candidate(top_candidates, &top_candidate_count, trimmed);
        }
        for (int j = 0; j < 3 && j < pages[i].line_count; j++) {
            int  idx = pages[i].line_count - 1 - j;
            char trimmed[MAX_LINE_LEN];
            strncpy(trimmed, pages[i].lines[idx], MAX_LINE_LEN - 1);
            trimmed[MAX_LINE_LEN - 1] = '\0';
            trim(trimmed);
            if (strlen(trimmed) > 0)
                add_candidate(bottom_candidates, &bottom_candidate_count, trimmed);
        }
    }

    /* ── Output cleaned text ── */
    for (int i = 0; i < page_count; i++) {
        for (int j = 0; j < pages[i].line_count; j++) {
            char trimmed[MAX_LINE_LEN];
            strncpy(trimmed, pages[i].lines[j], MAX_LINE_LEN - 1);
            trimmed[MAX_LINE_LEN - 1] = '\0';
            trim(trimmed);   /* FIX: trim() now modifies the buffer in-place */

            if (strlen(trimmed) == 0) continue;

            /* Skip header/footer lines */
            int is_hf = 0;
            if (j < 3)
                is_hf = is_header_footer(top_candidates, top_candidate_count, trimmed, page_count);
            if (!is_hf && j >= pages[i].line_count - 3)
                is_hf = is_header_footer(bottom_candidates, bottom_candidate_count, trimmed, page_count);
            if (is_hf) continue;

            /* Skip pure page-number lines (digits and whitespace only) */
            int all_digit = 1;
            for (int k = 0; trimmed[k]; k++) {
                if (!isdigit((unsigned char)trimmed[k]) && !isspace((unsigned char)trimmed[k])) {
                    all_digit = 0;
                    break;
                }
            }
            if (all_digit) continue;

            /* FIX: print the trimmed copy, not the original raw line */
            printf("%s\n", trimmed);
        }
    }

    return 0;
}