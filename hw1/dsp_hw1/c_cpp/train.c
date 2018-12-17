#include "hmm.h"
#include "myhead.h"
#include <math.h>

/**
 * Calculate alpha by forward algorithm
 * @param hmm model
 * @param observation
 * @param alpha
 */
void forward_algo(HMM *hmm, Observation *observ, Table *alpha)
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

    return;
}

/**
 * Calculate beta by backward algorithm
 * @param hmm model
 * @param observation
 * @param beta
 */
void backward_algo(HMM *hmm, Observation *observ, Table *beta)
{
    int i, j, t;
    beta->seq_num = observ->seq_num;
    beta->state_num = hmm->state_num;

    // Initialization
    for (i = 0; i < hmm->state_num; i++) {
        beta->table[observ->seq_num-1][i] = 1; // beta[T][i] = 1
    }

    // Induction
    for (t = observ->seq_num - 2; t >= 0; t--) {
        for (i = 0; i < hmm->state_num; i++) {
            double accum = 0;

            for (j = 0; j < hmm->state_num; j++) {
                accum += hmm->transition[i][j] * hmm->observation[observ->seq[t+1]][j] * beta->table[t+1][j];
            }

            beta->table[t][i] = accum; // beta[t][i] = \sum{a[i][j] * b[t+1][j] * beta[t+1][j]}
        }
    }

    return;
}

/**
 * Calculate delta by multiply alpha and beta
 * @param alpha
 * @param beta
 * @param delta
 */
void calc_delta(Table *alpha, Table *beta, Table *delta)
{
    int i, t;
    double sum;
    delta->seq_num = alpha->seq_num;
    delta->state_num = alpha->state_num;

    for (t = 0; t < delta->seq_num; t++) {
        sum = 0;
        for (i = 0; i < delta->state_num; i++) {
            delta->table[t][i] = alpha->table[t][i] * beta->table[t][i];
            sum += delta->table[t][i];
        }

        // Normalization
        for (i = 0; i < delta->state_num; i++) {
            delta->table[t][i] /= sum;
        }
    }

    return;
}

void baum_welch_algo(HMM *hmm, Observation *observ, Table *alpha, Table *beta, Epsilon *epsilon)
{
    int i, j, t;
    double sum;
    epsilon->seq_num = alpha->seq_num;
    epsilon->state_num = alpha->state_num;

    for (t = 0; t < epsilon->seq_num - 1; t++) {
        sum = 0;
        for (i = 0; i < epsilon->state_num; i++) {
            for (j = 0; j < epsilon->state_num; j++) {
                epsilon->table[t][i][j] = alpha->table[t][i] * \
                    hmm->transition[i][j] * \
                    hmm->observation[observ->seq[t+1]][j] * \
                    beta->table[t+1][j];
                
                sum += epsilon->table[t][i][j];
            }

        }

        // Normalization
        for (i = 0; i < epsilon->state_num; i++) {
            for (j = 0; j < epsilon->state_num; j++) {
                epsilon->table[t][i][j] /= sum;
            }
        }
    }
}

void train_model(HMM *hmm, Table *delta, Epsilon *epsilon, Observation *train, int train_num)
{
    int i, j, k, t, n;
    double sum_delta, sum_epsilon;
    double numerator, denominator;

    // update initial pi[i]
    for (i = 0; i < hmm->state_num; i++) {
        sum_delta = 0;
        for (n = 0; n < train_num; n++) {
            sum_delta += delta[n].table[0][i];
        }
        sum_delta /= train_num;
        hmm->initial[i] = sum_delta;
    }

    // update transition a[i][j]
    for (i = 0; i < hmm->state_num; i++) {
        for (j = 0; j < hmm->state_num; j++) {
            sum_delta = 0;
            sum_epsilon = 0;
            for (n = 0; n < train_num; n++) {
                for (t = 0; t < delta[n].seq_num - 1; t++) {
                    sum_epsilon += epsilon[n].table[t][i][j];
                    sum_delta += delta[n].table[t][i];
                }
            }
            hmm->transition[i][j] = sum_epsilon / sum_delta;
        }
    }

    // update observation b[k][j]
    for (k = 0; k < hmm->observ_num; k++) {
        for (j = 0; j < hmm->state_num; j++) {
            denominator = 0;
            numerator = 0;
            for (n = 0; n < train_num; n++) {
                for (t = 0; t < delta[n].seq_num; t++) {
                    if (train[n].seq[t] == k) {
                        numerator += delta[n].table[t][j];
                    }
                    denominator += delta[n].table[t][j];
                }
            }
            hmm->observation[k][j] = numerator / denominator;
        }
    }
}

Observation train[MAX_TRAIN_LINE];
Table alpha[MAX_TRAIN_LINE], beta[MAX_TRAIN_LINE], delta[MAX_TRAIN_LINE];
Epsilon epsilon[MAX_TRAIN_LINE];

int main(int argc, char *argv[])
{
    if (argc != 4 + 1) {
        printf("Wrong argument format\n");
        printf("Usage: ./train iteration model_init.txt seq_model_0X.txt model_0X.txt\n");
        exit(1);
    }

    int i, n, train_num;
    char *ptr;

    const int iter = strtol(argv[1], &ptr, 10);
    const char *model_init = argv[2];
    const char *train_file = argv[3];
    const char *model_file = argv[4];

    HMM hmm_initial;
    loadHMM(&hmm_initial, model_init);
    dumpHMM(stderr, &hmm_initial);

    train_num = get_data(train, train_file);

    for (i = 0; i < iter; i++) {
        printf("\n##### iteration: %d #####\n", i + 1);
        for (n = 0; n < train_num; n++) {
            forward_algo(&hmm_initial, &train[n], &alpha[n]);
            backward_algo(&hmm_initial, &train[n], &beta[n]);
            calc_delta(&alpha[n], &beta[n], &delta[n]);
            baum_welch_algo(&hmm_initial, &train[n], &alpha[n], &beta[n], &epsilon[n]);
        }
        train_model(&hmm_initial, delta, epsilon, train, train_num);
        dumpHMM(stderr, &hmm_initial);
    }

    printf("Dump HMM model to file: %s\n", model_file);
    FILE *fp = open_or_die(model_file, "w");
    dumpHMM(fp, &hmm_initial);
    fclose(fp);

    return 0;
}
