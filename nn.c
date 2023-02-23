#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BUFFER 1024*1024

#define sigmoid(x) (1 / (1 + exp(-x)))
#define dSigmoid(x) (x * (1 - x))
#define relu(x) ((x > 0) ? x : 0)
#define dRelu(x) ((x > 0) ? 1 : 0)
#define randomValue() ((double)rand())/((double)RAND_MAX)

struct NeuralNetwork {
  int numInputs;
  int numOutputs;
  int numLayers;
  int numEpochs;
  int numSets;
  int layerSizes[128];
  double trainingIn[10000][8];
  double trainingOut[10000][8];
};

void loadData(char *path, struct NeuralNetwork *nn);
void askParams(struct NeuralNetwork *nn);
int train(char *path, struct NeuralNetwork nn);
void shuffle(int *array, int size);

int main(int argc, char **argv)
{
  char *path = argv[1];
  struct NeuralNetwork nn;

  loadData(path, &nn);
  printf("Detected %d input(s) and %d output(s)\n", nn.numInputs, nn.numOutputs);
  askParams(&nn);
  train("model.ml", nn);
  return 0;
}

void loadData(char *path, struct NeuralNetwork *nn)
{
  int j=0, k=0, l=0, x=0;
  char file[BUFFER], aux[12];
  FILE *fp = fopen(path, "r");

  if (fp == NULL) {
    printf("Error opening dataset file!\n");
    exit(1);
  }

  fread(file, BUFFER, 1, fp);
  for (int i=0; j < 2 && file[i] != '\0'; i++) {
    for (x=0; aux[x] != '\0'; x++) aux[x] = '\0'; x=0;
    while ((file[i] >= '0' && file[i] <= '9') || file[i] == '.') {
      aux[x] = file[i]; i++; x++;
    }
    if (j == 0) nn->trainingIn[k][l] = atof(aux);
    else nn->trainingOut[k][l] = atof(aux);
//    printf("dataset[%d][%d][%d] = %lf\n", j, k, l, nn->dataset[j][k][l]);
    if (file[i] == ',') { l++; }
    else if (file[i] == ' ') { k++; l=0; }
    else if (file[i] == '\n' || file[i] == '\0') {
      if (j == 0) { nn->numInputs = l+1; }
      if (j == 1) { nn->numOutputs = l+1; nn->numSets = k+1; }
      j++; k=0; l=0;
    }
  }
}

void askParams(struct NeuralNetwork *nn)
{
  int layerSize;

  do {
    printf("Number of epochs : ");
    scanf(" %d", &nn->numEpochs);
  } while (nn->numEpochs < 1);

  // Multilayer network not implemented yet
  /*
  do {
    printf("Number of layers : ");
    scanf(" %d", &nn->numLayers);
  } while (nn->numLayers < 1);
  */
  nn->numLayers = 1;

  for (int i=0; i < nn->numLayers; i++) {
    printf("Size of layer %d : ", i);
    scanf(" %d", &layerSize);
    nn->layerSizes[i] = layerSize;
  }
}

void shuffle(int *array, int size) {
  for (int i = size - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int aux = array[i];
    array[i] = array[j];
    array[j] = aux;
  }
}

int train(char *path, struct NeuralNetwork nn) {

  // weigths and biases
  double hiddenLayer[nn.layerSizes[0]];
  double outputLayer[nn.numOutputs];
  double hiddenWeights[nn.numInputs][nn.layerSizes[0]];
  double outputWeights[nn.layerSizes[0]][nn.numOutputs];
  double hiddenLayerBias[nn.layerSizes[0]];
  double outputLayerBias[nn.numOutputs];

  // dataset
  int trainingSetOrder[nn.numSets];
  for (int i = 0; i < nn.numSets; i++) trainingSetOrder[i] = i;

  // deltas
  double deltaOutput[nn.numOutputs];
  double deltaHidden[nn.layerSizes[0]];

  // other
  double efficiency, lr = 0.1;
  char *file = path;

//  // check if model can be saved
//  FILE *fp = fopen(file, "w");
//  if (fp == NULL) {
//    printf("\nError opening save file!\n\n");
//    return 1;
//  }

  // init weigths between inputs and first layer
  printf("\n");
  for (int i=0; i < nn.numInputs; i++) {
    for (int j=0; j < nn.layerSizes[0]; j++) {
      hiddenWeights[i][j] = randomValue();
    }
  }
  // init biases between inputs and first layer
  for (int i=0; i < nn.layerSizes[0]; i++) {
    hiddenLayerBias[i] = randomValue();
    // init weigths between the first layer and outputs
    for (int j=0; j < nn.numOutputs; j++) {
      outputWeights[i][j] = randomValue();
    }
  }
  // init weigths between first layer and outputs
  for (int i=0; i < nn.numOutputs; i++) {
    outputLayerBias[i] = randomValue();
  }

  // train the model
  for(int epoch=0; epoch <= nn.numEpochs; epoch++) {
    // loop through a shuffled dataset
    shuffle(trainingSetOrder, nn.numSets);
    for (int x=0; x < nn.numSets; x++) {
      int i = trainingSetOrder[x];

      // compute hidden layer activations
      for (int j=0; j < nn.layerSizes[0]; j++) {
        double activation = hiddenLayerBias[j];
        for (int k=0; k < nn.numInputs; k++) {
          activation += nn.trainingIn[i][k] * hiddenWeights[k][j];
        }
        hiddenLayer[j] = sigmoid(activation);
      }

      // compute output layer activations
      for (int j=0; j < nn.numOutputs; j++) {
        double activation = outputLayerBias[j];
        for (int k=0; k < nn.layerSizes[0]; k++) {
          activation += hiddenLayer[k] * outputWeights[k][j];
        }
        outputLayer[j] = sigmoid(activation);
      }

      if (x == 0 && epoch % (nn.numEpochs/20) == 0) {
        printf ("Epoch : %d\t%lf > %lf\n", epoch, nn.trainingOut[i][0], outputLayer[0]);
      }

      // compute changes in output weights
      for (int j=0; j < nn.numOutputs; j++) {
        double errorOutput = (nn.trainingOut[i][j] - outputLayer[j]);
        deltaOutput[j] = errorOutput * dRelu(outputLayer[j]);
      }

      // compute changes in hidden weights
      for (int j=0; j < nn.layerSizes[0]; j++) {
        double errorHidden = 0;
        for(int k=0; k < nn.numOutputs; k++) {
          errorHidden += deltaOutput[k] * outputWeights[j][k];
        }
        deltaHidden[j] = errorHidden * dSigmoid(hiddenLayer[j]);
      }

      // apply changes in output weights
      for (int j=0; j < nn.numOutputs; j++) {
        outputLayerBias[j] += deltaOutput[j] * lr;
        for (int k=0; k < nn.layerSizes[0]; k++) {
          outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
        }
      }

      // apply changes in hidden weights
      for (int j=0; j < nn.layerSizes[0]; j++) {
        hiddenLayerBias[j] += deltaHidden[j] * lr;
        for(int k=0; k < nn.numInputs; k++) {
          hiddenWeights[k][j] += nn.trainingIn[i][k] * deltaHidden[j] * lr;
        }
      }

    }
  }

  printf("\n");
  double inputNeuron[nn.numInputs];

  while(1) {

    for(int i=0; i < nn.numInputs; i++) {
      printf("Input neuron %d : ", i);
      scanf(" %lf", &inputNeuron[i]);
    }

    // compute hidden layer activations
    for (int j=0; j < nn.layerSizes[0]; j++) {
      double activation = hiddenLayerBias[j];
      for (int k=0; k < nn.numInputs; k++) {
        activation += inputNeuron[k] * hiddenWeights[k][j];
      }
      hiddenLayer[j] = sigmoid(activation);
    }

    // compute output layer activations
    for (int j=0; j < nn.numOutputs; j++) {
      double activation = outputLayerBias[j];
      for (int k=0; k < nn.layerSizes[0]; k++) {
        activation += hiddenLayer[k] * outputWeights[k][j];
      }
      outputLayer[j] = sigmoid(activation);
      printf ("> Output layer %d : %g\n", j, outputLayer[j]);
    }

  }

//  // save model info
//  fprintf(fp, "{\"inputs\":%d,\"outputs\":%d,\"hiddenNodes\":%d,",
//    nn.numInputs, nn.numOutputs, nn.layerSizes[0]);
//
//  // save weigths between inputs and first layer
//  fprintf(fp, "\"hiddenWeights\":[");
//  for (int i=0; i < nn.numInputs; i++) {
//    for (int j=0; j < nn.layerSizes[0]; j++) {
//      if (i+1 == nn.numInputs && j+1 == nn.layerSizes[0])
//        fprintf(fp, "%lf],", hiddenWeights[i][j]);
//      else fprintf(fp, "%lf,", hiddenWeights[i][j]);
//    }
//  }
//
//  // save weigths between the first layer and outputs
//  fprintf(fp, "\"outputWeights\":[");
//  for (int i=0; i < nn.layerSizes[0]; i++) {
//    for (int j=0; j < nn.numOutputs; j++) {
//      if (i+1 == nn.layerSizes[0] && j+1 == nn.numOutputs)
//        fprintf(fp, "%lf],", outputWeights[i][j]);
//      else fprintf(fp, "%lf,", outputWeights[i][j]);
//    }
//  }
//
//  // save biases between inputs and first layer
//  fprintf(fp, "\"hiddenBiases\":[");
//  for (int i=0; i < nn.layerSizes[0] - 1; i++) {
//    fprintf(fp, "%lf,", hiddenLayerBias[i]);
//  }
//  fprintf(fp, "%lf],", hiddenLayerBias[nn.layerSizes[0]-1]);
//
//  // save weigths between first layer and outputs
//  fprintf(fp, "\"outputBiases\":[");
//  for (int i=0; i < nn.numOutputs - 1; i++) {
//    fprintf(fp, "%lf,", outputLayerBias[i]);
//  }
//  fprintf(fp, "%lf]}", outputLayerBias[nn.numOutputs-1]);
//
//  printf ("\nModel saved at %s!\n\n", file);
//  fclose(fp);
  return 0;

}
