#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <math.h>

#define MB 1024*1024
#define sigmoid(x) (1 / (1 + exp(-x)))
#define dSigmoid(x) (x * (1 - x))
#define relu(x) ((x > 0) ? x : 0)
#define dRelu(x) ((x > 0) ? 1 : 0)
#define randomValue() ((double)rand())/((double)RAND_MAX)

struct HyperParams {
  size_t numInputs;
  size_t numOutputs;
  size_t numSets;
  size_t numEpochs;
  size_t numLayers;
  size_t *layerSizes;
  float learningRate;
};

struct Neurons {
  float (*dataset)[1][1];
  float (*weights)[1][1];
  float **biases;
  float **values;
};

int detectMode(char *path) {
  int i=0;
  for (; path[i+2] != '\0'; i++);
  if (path[i] == 'd' && path[i+1] == 's') return 1;
  if (path[i] == 'm' && path[i+1] == 'l') return 0;
  else {
    printf("ERROR : Unrecognized file extension\n");
    exit(1);
  }
}

int parseFile(FILE* fp, char **data, size_t *size, struct HyperParams *hp) {
  fseek(fp, 0L, SEEK_END); *size = ftell(fp); rewind(fp);
  *data = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fileno(fp), 0);
  if (*data == MAP_FAILED) { perror("Not enough memory"); exit(1); }

  size_t i=0;
  hp->numInputs = 1; hp->numOutputs = 1; hp->numSets = 1;
  for (; (*data)[i] != ' ' && i < *size; i++)
    if ((*data)[i] == ',') hp->numInputs++;
  for (; (*data)[i] != '\n' && i < *size; i++)
    if ((*data)[i] == ' ') hp->numSets++;
  for (; (*data)[i] != ' ' && i < *size; i++)
    if ((*data)[i] == ',') hp->numOutputs++;

  return 0;
}

size_t loadData(char *data, size_t size, struct Neurons *n, struct HyperParams *hp) {
  size_t toAlloc = sizeof(float) * hp->numSets * (hp->numInputs + hp->numOutputs);
  size_t line=0, set=0, value=0, i=0, j=0;
  char tmp[8];

  n->dataset = malloc(toAlloc);
  for (; i < size; i++) {
    switch(data[i]) {
      case ',': value++; break;
      case ' ': set++; value=0; break;
      case '\n': line++; set=0; value=0; break;
      default:
        while (data[i] >= '0' && data[i] <= '9' || data[i] == '.') {
          tmp[j] = data[i]; i++; j++;
        }
        tmp[j] = '\0'; j=0; i--;
        n->dataset[line][set][value] = atof(tmp);
        // printf("n->dataset[%zu][%zu][%zu] = %f\n", line, set, value, atof(tmp));
      break;
    }
  }

  munmap(data, size);
  return toAlloc;
}

void train(FILE *fp) {
  char *data;
  size_t size;
  struct Neurons n;
  struct HyperParams hp;

  parseFile(fp, &data, &size, &hp);
  printf("Detected %zu sets, %zu inputs, %zu outputs\n", hp.numSets, hp.numInputs, hp.numOutputs);
  printf("Successfully loaded the dataset (%.2f MB)\n", loadData(data, size, &n, &hp) / (1.0*MB));
  printf("\n");
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage : %s <dataset.ds>|<model.ml>\n", argv[0]);
    return 1;
  }

  char *path = argv[1];
  FILE *fp = fopen(path, "r");
  if (fp == NULL) {
    printf("ERROR : File doesn't exist\n");
    exit(1);
  }

  if (detectMode(path)) {
    printf("\nTraining using %s\n", path);
    train(fp);
  } else {
    printf("\nRunning %s\n", path);
    //run(path);
  }

  return 0;
}
