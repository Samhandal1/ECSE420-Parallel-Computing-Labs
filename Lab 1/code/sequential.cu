#include <stdio.h>
#include <stdlib.h>

int logic_gate(int A, int B, int gate) {
    switch (gate) {
        case 0: return A && B;        // AND
        case 1: return A || B;        // OR
        case 2: return !(A && B);     // NAND
        case 3: return !(A || B);     // NOR
        case 4: return A ^ B;         // XOR
        case 5: return !(A ^ B);      // XNOR
        default: return -1;           // Invalid gate type
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: ./sequential <input_file_path> <input_file_length> <output_file_path>\n");
        return 1;
    }

    FILE *input_file = fopen(argv[1], "r");
    int N = atoi(argv[2]);
    FILE *output_file = fopen(argv[3], "w");

    for (int i = 0; i < N; i++) {
        int A, B, gate;
        fscanf(input_file, "%d,%d,%d", &A, &B, &gate);
        fprintf(output_file, "%d\n", logic_gate(A, B, gate));
    }

    fclose(input_file);
    fclose(output_file);

    return 0;
}
