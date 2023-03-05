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
  size_t numParams;
  size_t numLayers;
  size_t *layerSizes;
  float learningRate;
};

struct Neurons {
  float (*dataset)[1][1];
  float (*weights)[1][1];
  float (*biases)[1];
  float (*values)[1];
};

void train(FILE *fp);
void askParams(struct HyperParams *hp);
void freeAll(struct Neurons *n, struct HyperParams *hp);
int detectMode(char *path);
int parseFile(FILE* fp, char **data, size_t *size, struct HyperParams *hp);
size_t loadData(char *data, size_t size, struct Neurons *n, struct HyperParams *hp);
size_t initNeurons(size_t *size, struct Neurons *n, struct HyperParams *hp);

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage : %s <dataset.ds>|<model.ml>\n", argv[0]);
    return 1;
  }

  char *path = argv[1];
  FILE *fp = fopen(path, "r");
  if (fp == NULL) {
    perror("ERROR : File doesn't exist\n");
    exit(1);
  }

  if (detectMode(path)) {
    printf("\nTraining using %s\n\n", path);
    train(fp);
  } else {
    printf("\nRunning %s\n\n", path);
    //run(path);
  }

  return 0;
}

int detectMode(char *path) {
  int i=0;
  for (; path[i+2] != '\0'; i++);
  if (path[i] == 'd' && path[i+1] == 's') return 1;
  if (path[i] == 'm' && path[i+1] == 'l') return 0;
  else {
    perror("ERROR : Unrecognized file extension\n");
    exit(1);
  }
}

void train(FILE *fp) {
  char *data;
  size_t size;
  struct Neurons n;
  struct HyperParams hp;

  askParams(&hp);
  parseFile(fp, &data, &size, &hp);
  printf("\nDetected %zu sets, %zu inputs, %zu outputs\n",
    hp.numSets, hp.numInputs, hp.numOutputs);
  printf("Successfully loaded the dataset (%.2f MB)\n",
    loadData(data, size, &n, &hp) / (1.0*MB));
  initNeurons(&size, &n, &hp);
  printf("Successfully initialized %zu parameters (%.2f MB)\n",
    hp.numParams, size / (1.0*MB));
  printf("\n");

  freeAll(&n, &hp);
}

void askParams(struct HyperParams *hp) {
  int layerSize;

  do {
    printf("Number of epochs : ");
    scanf(" %d", &hp->numEpochs);
  } while (hp->numEpochs < 1);

  do {
    printf("Number of layers : ");
    scanf(" %d", &hp->numLayers);
  } while (hp->numLayers < 1);
  hp->numLayers += 2;
  hp->layerSizes = malloc(sizeof(float *) * hp->numLayers);

  for (int i=1; i < hp->numLayers-1; i++) {
    printf("Size of layer %d : ", i);
    scanf(" %d", &layerSize);
    hp->layerSizes[i] = layerSize;
  }
}

int parseFile(FILE* fp, char **data, size_t *size, struct HyperParams *hp) {
  fseek(fp, 0L, SEEK_END); *size = ftell(fp); rewind(fp);
  *data = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fileno(fp), 0);
  if (*data == MAP_FAILED) { perror("ERROR : Not enough memory"); exit(1); }

  size_t i=0;
  hp->numInputs = 1; hp->numOutputs = 1; hp->numSets = 1;
  for (; (*data)[i] != ' ' && i < *size; i++)
    if ((*data)[i] == ',') hp->numInputs++;
  for (; (*data)[i] != '\n' && i < *size; i++)
    if ((*data)[i] == ' ') hp->numSets++;
  for (; (*data)[i] != ' ' && i < *size; i++)
    if ((*data)[i] == ',') hp->numOutputs++;

  hp->layerSizes[0] = hp->numInputs;
  hp->layerSizes[hp->numLayers-1] = hp->numOutputs;
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

size_t initNeurons(size_t *size, struct Neurons *n, struct HyperParams *hp) {
  size_t i, j, k, neuronCount = 0, weightCount = 0;

  for (i=0; i < hp->numLayers; i++) {
    neuronCount += hp->layerSizes[i];
    if (i+1 < hp->numLayers)
      weightCount += hp->layerSizes[i] * hp->layerSizes[i+1];
  }

  *size = sizeof(float) * ((neuronCount * 2) + weightCount);
  hp->numParams = neuronCount + weightCount - hp->layerSizes[0];
  n->biases = malloc(sizeof(float) * neuronCount);
  n->values = malloc(sizeof(float) * neuronCount);
  n->weights = malloc(sizeof(float) * weightCount);

  if (n->biases == NULL || n->values == NULL || n->weights == NULL) {
    perror("ERROR : Not enough memory");
    exit(1);
  }

  for (i=0; i+1 < hp->numLayers; i++)
    for (j=0; j < hp->layerSizes[i]; j++)
      for (k=0; k < hp->layerSizes[i+1]; k++)
        n->weights[i][j][k] = randomValue();
        // printf("n->weights[%zu][%zu][%zu] = %f\n", i, j, k, n->weights[i][j][k]);

  for (i=0; i < hp->numLayers; i++)
    for (j=0; j < hp->layerSizes[i]; j++)
      n->biases[i][j] = randomValue();
      // printf("n->biases[%zu][%zu] = %f\n", i, j, n->biases[i][j]);

  for (i=0; i < hp->numLayers; i++)
    for (j=0; j < hp->layerSizes[i]; j++)
      n->values[i][j] = randomValue();
      // printf("n->values[%zu][%zu] = %f\n", i, j, n->values[i][j]);

  return 0;
}

void freeAll(struct Neurons *n, struct HyperParams *hp) {
  free(hp->layerSizes);
  free(n->dataset);
  free(n->weights);
  free(n->biases);
  free(n->values);
}
