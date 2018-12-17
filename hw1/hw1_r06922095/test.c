#include "hmm.h"
#include "myhead.h"
#include <math.h>

/**
 * @param hmm model
 * @param observation
 * @param delta
 * @param psi
 * @return likelihood on the observation given hmm model
 */
double viterbi_algo(HMM *hmm, Observation *observ, Table *delta, Table *psi)
{
    int i, j, t, arg_max;
    double max, tmp;
    delta->seq_num = observ->seq_num;
    delta->state_num = hmm->state_num;
    psi->seq_num = observ->seq_num;
    psi->state_num = hmm->state_num;

    // Initialization
    for (i = 0; i < hmm->state_num; i++) {
        delta->table[0][i] = hmm->initial[i] * hmm->observation[observ->seq[0]][i]; // delta[0][i] = pi[i] * b[o_1][i]
    }

    // Recursion
    for (t = 0; t < observ->seq_num - 1; t++) {
        for (j = 0; j < hmm->state_num; j++) {
            max = 0;

            for (i = 0; i < hmm->state_num; i++) {
                tmp = delta->table[t][i] * hmm->transition[i][j];
                if (tmp > max) {
                    arg_max = i;
                    max = tmp;
                }
            }

            delta->table[t+1][j] = max * hmm->observation[observ->seq[t+1]][j]; // delta[t][j] = \sum{delta[t][i] * a[i][j]} * b[t][j]
            psi->table[t+1][j] = arg_max;
        }
    }

    // Termination
    double p = 0;
    Observation q;
    q.seq_num = observ->seq_num;

    for (i = 0; i < hmm->state_num; i++) {
        if (delta->table[observ->seq_num-1][i] > p) {
            p = delta->table[observ->seq_num-1][i];
            q.seq[observ->seq_num-1] = i;
        }
    }

    // Path backtracking
    for (t = observ->seq_num - 2; t >= 0; t--) {
        q.seq[t] = psi->table[t+1][q.seq[t+1]]; // q[t] = psi[t+1][q[t+1]]
    }

    return p;
}

/**
 * Calculate alpha by forward algorithm
 * @param hmm model
 * @param observation
 * @param alpha
 */
double forward_algo(HMM *hmm, Observation *observ, Table *alpha)
{
    int i, j, t;
    alpha->seq_num = observ->seq_num;
    alpha->state_num = hmm->state_num;

    // Initialization
    for (i = 0; i < hmm->state_num; i++) {
        alpha->table[0][i] = hmm->initial[i] * hmm->observation[observ->seq[0]][i]; // alpha[0][i] = pi[i] * b[o_1][i]
    }

    // Induction
    for (t = 0; t < observ->seq_num - 1; t++) {
        for (j = 0; j < hmm->state_num; j++) {
            double accum = 0;

            for (i = 0; i < hmm->state_num; i++) {
                accum += alpha->table[t][i] * hmm->transition[i][j];
            }

            alpha->table[t+1][j] = accum * hmm->observation[observ->seq[t+1]][j]; // alpha[t][j] = \sum{alpha[t][i] * a[i][j]} * b[t][j]
        }
    }

    // Termination
    double prob = 0;
    for (i = 0; i < hmm->state_num; i++) {
        prob += alpha->table[observ->seq_num-1][i];
    }

    return prob;
}

Observation test[MAX_TEST_LINE];

int main(int argc, char *argv[])
{
    if (argc != 3 + 1) {
        printf("Wrong argument format\n");
        printf("Usage: ./test modellist.txt testing_data.txt result.txt\n");
        exit(1);
    }

    int i, j, test_num, arg_max;
    double prob, max;
    int pred[MAX_TEST_LINE];
    double likelihood[MAX_TEST_LINE];
    Table alpha, delta, psi;

    const char *modellist = argv[1];
    const char *test_file = argv[2];
    const char *result_file = argv[3];

    HMM hmms[5];
    load_models(modellist, hmms, 5);
    dump_models(hmms, 5);

    test_num = get_data(test, test_file);

    for (i = 0; i < test_num; i++) {
        max = 0;
        for (j = 0; j < 5; j++) {
            // choose one: use forward algo. or viterbi algo.
            prob = forward_algo(&hmms[j], &test[i], &alpha);
            // prob = viterbi_algo(&hmms[j], &test[i], &delta, &psi);
            if (prob > max) {
                max = prob;
                arg_max = j;
            }
        }
        pred[i] = arg_max;
        likelihood[i] = max;
    }

    printf("Dump result to file: %s\n", result_file);
    FILE *fp = open_or_die(result_file, "w");
    for (i = 0; i < test_num; i++) {
        fprintf(fp, "%s ", hmms[pred[i]].model_name);
        fprintf(fp, "%e\n", likelihood[i]);
    }
    fclose(fp);
    
    return 0;
}
