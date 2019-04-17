#include <math.h>
#include <stdlib.h>
#include <stdio.h> 
#include "map.h"
#include <time.h>
#include<windows.h>


typedef struct layer_ {
	double **param_W;
	double *param_b;
	int activation;
	int node_num;
	int input_num;
	double *Z;
	double *A;
	double **dw;
	double *db;
}layer;

typedef struct network_ {
	layer *layer;
	int layer_num;
}network;

typedef struct training_temp_ {
	double accuracy_history;
	double cost;
}training_temp;

double sigmoid(double z) {
	return 1 / (1 + exp(-z));
}

double relu(double z) {
	return max(0, z);
}

double sigmoid_backward(double z) {
	double sig = sigmoid(z);
	return sig*(1 - sig);
}

double relu_backward(double z) {
	if (z > 0) {
		return 1;
	}
	else
	{
		return 0;
	}
}

double randn(double mu, double sigma){
	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;
	srand((unsigned)time(NULL));
	if (call == 1)
	{
		call = !call;
		return (mu + sigma * (double)X2);
	}

	do
	{
		U1 = -1 + ((double)rand() / RAND_MAX) * 2;
		U2 = -1 + ((double)rand() / RAND_MAX) * 2;
		W = pow(U1, 2) + pow(U2, 2);
	} while (W >= 1 || W == 0);

	mult = sqrt((-2 * log(W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double)X1);
}


void dot(double *input, layer *l,double *result1){
	for (int i = 0; i < l->node_num; i++) {
		double temp = 0;
		double x = 0;
		for (int j = 0; j < l->input_num; j++) {
			x = l->param_W[i][j] * input[j];
			temp += x;
			x = 0;
		}
		result1[i] = temp;
	}

	for (int i = 0; i < l->node_num; i++) {
		result1[i] += l->param_b[i];
		l->Z[i] = result1[i];
	}
	
	if (l->activation == 1){
		for (int i = 0; i < l->node_num; i++) {
			result1[i] = relu(result1[i]);
			l->A[i] = result1[i];
		}
	}
	else if (l->activation == 0){
		for (int i = 0; i < l->node_num; i++) {
			result1[i] = sigmoid(result1[i]);
			l->A[i] = result1[i];
		}
	}
	for (int i = 0; i < l->node_num; i++) {
		input[i] = result1[i];
	}
}

double * forward_propagation(double *input, network *mynet) {
	double *temp = (double *)malloc(20 * sizeof(double));
	double *yue = (double *)malloc(20 * sizeof(double));
	yue[0] = input[0];
	yue[1] = input[1];
	yue[2] = input[2];
	for (int i = 0; i < mynet->layer_num; i++) {
		dot(yue, &(mynet->layer[i]), temp);
	}
	return temp;
}

void dot_(double *input, layer *l, double *result1) {
	for (int i = 0; i < l->node_num; i++) {
		double temp = 0;
		double x = 0;
		for (int j = 0; j < l->input_num; j++) {
			x = l->param_W[i][j] * input[j];
			temp += x;
			x = 0;
		}
		result1[i] = temp;
	}

	for (int i = 0; i < l->node_num; i++) {
		result1[i] += l->param_b[i];
	}

	if (l->activation == 1) {
		for (int i = 0; i < l->node_num; i++) {
			result1[i] = relu(result1[i]);
		}
	}
	else if (l->activation == 0) {
		for (int i = 0; i < l->node_num; i++) {
			result1[i] = sigmoid(result1[i]);
		}
	}
	for (int i = 0; i < l->node_num; i++) {
		input[i] = result1[i];
	}
}

double * forward_propagation_(double *input, network *mynet) {
	double *temp = (double *)malloc(10 * sizeof(double));
	double *yue = (double *)malloc(10 * sizeof(double));
	yue[0] = input[0];
	yue[1] = input[1];
	yue[2] = input[2];
	for (int i = 0; i < mynet->layer_num; i++) {
		dot_(yue, &(mynet->layer[i]), temp);
	}
	return temp;
}


double * dot_de_din(double *result,double *de_dout,double *dout_din,int num_node) {
	for (int i = 0; i < num_node; i++) {
		result[i] = de_dout[i] * dout_din[i];
	}
	return result;
}

/*行数是本层节点个数，列数是上层节点个数*/
void dot_de_dw(double *de_din, layer *lp, int num_node,int num_node_pre,layer *l) {
	for (int i = 0; i < num_node; i++) {
		for (int j = 0; j < num_node_pre; j++) {
			l->dw[i][j] = de_din[i] * lp->A[j];
		}
	}
}
void dot_de_dw_(double *de_din,double *input, int num_node, int num_node_pre, layer *l) {
	for (int i = 0; i < num_node; i++) {
		for (int j = 0; j < num_node_pre; j++) {
			l->dw[i][j] = de_din[i] * input[j];
		}
	}
}

double * dot_da_prev(double *res, double *de_din, layer l, int num_node, int num_node_pre) {
	for (int i = 0; i < num_node_pre; i++) {
		double sum = 0;
		for (int j = 0; j < num_node; j++) {
			sum += de_din[j] * l.param_W[j][i];
		}
		res[i] = sum;
	}
	return res;
}

double * single_backwards(double *da_cur,layer *l,layer *lp) {
	double *result = (double *)malloc(20 * sizeof(double));
	if (l->activation == 1) {
		for (int i = 0; i < l->node_num; i++) {
			result[i] = relu_backward(l->Z[i]);
		}
	}
	else if (l->activation == 0) {
		for (int i = 0; i < l->node_num; i++) {
			result[i] = sigmoid_backward(l->Z[i]);
		}
	}
	double *result1 = (double *)malloc(20 * sizeof(double));
	dot_de_din(result1, da_cur, result, l->node_num);

	for (int i = 0; i < l->node_num; i++) {
		l->db[i] = result1[i];
	}
	dot_de_dw(result1, lp, l->node_num,lp->node_num, l);

	double *result2 = (double *)malloc(20 * sizeof(double));
	dot_da_prev(result2, result1, *l, l->node_num, lp->node_num);
	free(result);
	free(result1);
	return result2;
}

double * single_backwards_layer0(double *da_cur, layer *l,double *inp) {
	double *result3 = (double *)malloc(20 * sizeof(double));
	if (l->activation == 1) {
		for (int i = 0; i < l->node_num; i++) {
			result3[i] = relu_backward(l->Z[i]);
		}
	}
	else if (l->activation == 0) {
		for (int i = 0; i < l->node_num; i++) {
			result3[i] = sigmoid_backward(l->Z[i]);
		}
	}
	double *result4 = (double *)malloc(20 * sizeof(double));
	dot_de_din(result4, da_cur, result3, l->node_num);
	for (int i = 0; i < l->node_num; i++) {
		l->db[i] = result4[i];
	}
/*
	l->dw = (double **)malloc(l->node_num * sizeof(double*));
	for (int i = 0; i < l->node_num; i++) {
		l->dw[i] = (double*)malloc(l->input_num * sizeof(double));
	}
*/
	dot_de_dw_(result4, inp, l->node_num, l->input_num, l);

	double *result5 = (double *)malloc(20 * sizeof(double));
	dot_da_prev(result5,result4, *l, l->node_num, l->input_num);
	free(result4);
	free(result3);
	return result5;
}

void full_backward(double *y_predict,network *mynet,double *result,double *input) {
	int output_num = mynet->layer[mynet->layer_num - 1].node_num;
	double *da_prev = (double *)malloc(20 * sizeof(double));
	double *da_curr;
	
	for (int i = 0; i <output_num ; i++) {
		da_prev[i] = -result[i] / y_predict[i] + (1 - result[i]) / (1 - y_predict[i]);
	}
	for (int i = mynet->layer_num - 1; i >= 0; i--) {
		if (i == 0) {
			da_curr = da_prev;
			da_prev = single_backwards_layer0(da_curr, &(mynet->layer[0]), input);
			break;
		}
		da_curr = da_prev;
		da_prev = single_backwards(da_curr, &(mynet->layer[i]), &(mynet->layer[i - 1]));
	}
	return;
}
/*TODO: 第一层怎么办？
db怎么做？
*/
void update(network *mynet,double learning_rate) {
	for (int i = 0; i < mynet->layer_num; i++) {
		for (int j = 0; j < mynet->layer[i].node_num; j++) {
			for (int k = 0; k < mynet->layer[i].input_num; k++) {
				mynet->layer[i].param_W[j][k] -= learning_rate * mynet->layer[i].dw[j][k];
			}
			mynet->layer[i].param_b[j] -= learning_rate * mynet->layer[i].db[j];
		}
	}
	return;
}

training_temp train(double **in, double *label, network *mynet, int epochs, double learning_rate,int input_len) {
	training_temp tt;
	FILE *fpcost = fopen("datacost.txt", "w");
	FILE *fpacc = fopen("dataacc.txt", "w");

	double *labeli = (double *)malloc(input_len * sizeof(double));
	double *test = (double *)malloc(3 * sizeof(double));
	double *result;
	tt.accuracy_history = 0;
	tt.cost = 0;
	for (int j = 0; j < input_len / epochs; j++) {
		int sum = 0;
		for (int i = j*epochs; i < (j+1)*epochs; i++) {
			/*printf("\n data in: %lf,%lf \n", in[i][0], in[i][1]);
			*/
			result = forward_propagation(in[i], mynet);
			/*printf("result: %lf,%lf \n", result[0], result[1]);
			*/
			int index;
			/*printf("label: %lf", label[i]);
			*/
			if (result[0] > result[1]) {
				index = 0;
			}
			else {
				index = 1;
			}
			if (index == label[i]) {
				sum += 1;
			}
			if (label[i] == 0){
				labeli[0] = 1; labeli[1] = 0;
			}
			else {
				labeli[0] = 0; labeli[1] = 1;
			}
			full_backward(result, mynet, labeli,in[i]);
			update(mynet,learning_rate);
			tt.cost = -labeli[0] * log(result[0]) - (1 - labeli[0])*log(1 - result[0]) - labeli[1] * log(result[1]) - (1 - labeli[1])*log(1 - result[1]);
			
		}
		tt.accuracy_history = sum / (double)epochs;
		fprintf(fpcost, "%lf\n", tt.cost);
		fprintf(fpacc, "%lf\n", tt.accuracy_history);
		printf("\n***************************accuracy***********************");
		printf("%lf ", tt.accuracy_history);
		printf("%lf", tt.cost);
		
	}

	fclose(fpcost);
	fclose(fpacc);
	return tt;
}




double ** init_layer_w(int *layer) {
	printf("\n参数W:");
	double **layer_param_W = (double **)malloc(layer[1]*sizeof(double*));
	for (int j = 0; j < layer[1]; j++) {
		printf("\n 节点%d: ", j);
		double *layer_param_w = (double *)malloc(layer[0] * sizeof(double));
		for (int i = 0; i < layer[0]; i++) {
			layer_param_w[i] = randn(2.5,3)*0.01;
			Sleep(50);
			printf("%lf ", layer_param_w[i]);
		}
		layer_param_W[j] = layer_param_w;
	}
	return layer_param_W;
}

double * init_layer_b(int *layer) {
	printf("\n参数b:");
	double *layer_param_b = (double *)malloc(layer[1] * sizeof(double));
	for (int i = 0; i < layer[1]; i++) {
		printf("\n 节点%d: ", i);
		Sleep(50);
		layer_param_b[i] = randn(2.5, 3)*0.01;
		printf("%lf ", layer_param_b[i]);
	}
	return layer_param_b;
}

layer init_layer(int *layer_Info) {
	layer this_layer ;
	this_layer.param_W = init_layer_w(layer_Info);
	this_layer.param_b = init_layer_b(layer_Info);
	this_layer.activation = layer_Info[2];
	this_layer.node_num = layer_Info[1];
	this_layer.input_num = layer_Info[0];
	this_layer.Z = (double *)malloc(this_layer.node_num * sizeof(double));
	this_layer.A = (double *)malloc(this_layer.node_num * sizeof(double));
	this_layer.db = (double *)malloc(this_layer.node_num * sizeof(double));
	this_layer.dw = (double **)malloc(this_layer.node_num * sizeof(double*));
	for (int i = 0; i < this_layer.node_num; i++) {
		this_layer.dw[i] = (double*)malloc(this_layer.input_num * sizeof(double));
	}
	
	
	return this_layer;
}

network * init(int **architecture,int n,int m) {
	network *net = (network *)malloc(sizeof(network));
	net->layer = (layer *)malloc(n * sizeof(layer));
	for (int i = 0; i < n; i++) {
		net->layer[i] = init_layer(architecture[i]);
	}
	net->layer_num = n;
	return net;
}


void test(int (*arch)[3]) {
	int x = arch[0][1];
}

double ** create_data(int num) {
	double **test_data = (double **)malloc(num * sizeof(double *));
	srand(150);
	for (int i = 0; i < num; i++) {
		test_data[i] = (double *)malloc(3 * sizeof(double));
	}
	for (int i = 0; i < num; i++) {
		double x = rand() % 100;
		double y = (rand() % 100);
		double z = (rand() % 100);
		test_data[i][0] = x;
		test_data[i][1] = y;
		test_data[i][2] = y;
	}

	
	return test_data;
}
void normalization(double **data,int num) {
	double maxx = 0;
	double maxy = 0;
	double maxz = 0;
	for (int i = 0; i < num; i++) {
		if (maxx < data[i][0]) {
			maxx = data[i][0];
		}
		if (maxy < data[i][1]) {
			maxy = data[i][1];
		}
		if (maxz < data[i][2]) {
			maxz = data[i][2];
		}
	}
	for (int i = 0; i < num; i++) {
		data[i][0] /= maxx;
		data[i][1] /= maxy;
		data[i][2] /= maxz;
	}	
}

double * creat_label(double **data,int num) {
	double *label = (double *)malloc(num * sizeof(double));
	for (int i = 0; i < num; i++) {
		if (pow(data[i][0],2)- 2*pow(data[i][1],2)+pow(data[i][2],2)> 0) {
			label[i] = 1;
		}else {
			label[i] = 0;
		}
	}
	return label;
}

void main() {
	int n = 1;
	int m = 3;

	double **data = create_data(50000);
	double *label = creat_label(data, 50000);
	normalization(data, 50000);
	int **arr = (int **)malloc(n * sizeof(int *));
	for (int i = 0; i < n; i++) {
		arr[i] = (int*)malloc(m*sizeof(int));
	}
	int l1[] = { 3 ,2, 0 };
	/*
	int l2[] = { 4, 6, 0 };
	int l3[] = { 6, 6, 0 };
	int l4[] = { 6, 4, 0 };
	*/
	int l5[] = { 2, 2, 0 };
	arr[0] = l1;
	arr[1] = l5;
	/*
	arr[2] = l3;
	arr[3] = l4;
	arr[4] = l5;
	*/
	network *mynetwork = init(arr, n, m);
	
	train(data,label,mynetwork,50,0.005, 50000);
	
	getchar();

}