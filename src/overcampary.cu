/*
 ============================================================================
 Name        : CAMPARYtest.cu
 Author      : amakje
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include "multi_prec.h"
#include <mpfr.h>
//#include <params.h>
#include <mpfloat.cuh>
#include <time.h>
#include <random>
#include <chrono>
#include <rns.cuh>
#include <gmp.h>

//#define CAMPARY_PRECISION 4

//Execution configuration
#define CAMPARY_VECTOR_MULTIPLY_THREADS 32




int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits


	//СОЗДАНИЕ МАССИВОВ
		mpfr_t * create_random_array(unsigned long size, unsigned long bits)
		{
	    srand(time(NULL));

	    gmp_randstate_t state;                           // Random generator state object
	    gmp_randinit_default(state);                     // Initialize state for a Mersenne Twister algorithm
	    gmp_randseed_ui(state, (unsigned) time(NULL));   // Call gmp_randseed_ui to set initial seed value into state

	    std::uniform_real_distribution<double> unif(-1, 1);
	    std::default_random_engine re;
	    re.seed(std::chrono::system_clock::now().time_since_epoch().count());

	    mpz_t random_number;
	    mpz_init2(random_number, bits);
	    mpfr_t pow_bits;
	    mpfr_init2(pow_bits, bits);
	    mpfr_set_d(pow_bits, 2, MPFR_RNDN);
	    mpfr_pow_si(pow_bits, pow_bits, -1 * bits, MPFR_RNDN);

	    mpfr_t* array = new mpfr_t[size];

	    for(int i = 0; i < size; i ++){
	        mpfr_init2(array[i], bits);
	    }

	    for (int i = 0; i < size; i++) {
	        mpz_urandomb(random_number, state, bits);
	        //Generate a uniformly distributed random double x
	        mpfr_set_z(array[i], random_number, MPFR_RNDD);
	        mpfr_mul_d(array[i], array[i], unif(re), MPFR_RNDN);
	        mpfr_mul(array[i], array[i], pow_bits, MPFR_RNDN);
	    }
	    return array;
	}
//КОНВЕРИТРОВАНИЕ
	std::string convert_to_string_sci(mpfr_t number, int ndigits)
	{
	    char * significand;
	    long exp = 0;
	    //Convert number to a string of digits in base 10
	    significand = mpfr_get_str(NULL, &exp, 10, ndigits, number, MPFR_RNDN);
	    //Convert to std::string
	    std::string number_string(significand);
	    //Set decimal point
	    if(number_string.compare(0, 1, "-") == 0){
	        number_string.insert(1, "0.");
	    }else {
	        number_string.insert(0, "0.");
	    }
	    //Add the exponent
	    number_string += "e";
	    number_string += std::to_string(exp);
	    //Cleanup
	    mpfr_free_str(significand);
	    return number_string;
	}
//ВЫДАЧА РЕЗУЛЬТАТА
	template<int nterms>
	static void printResult(multi_prec<nterms> result){
	    int p = 8192;
	    mpfr_t x;
	    mpfr_t r;
	    mpfr_init2(x, p);
	    mpfr_init2(r, p);
	    mpfr_set_d(r, 0.0, MPFR_RNDN);
	    for(int i = nterms - 1; i >= 0; i--){
	        mpfr_set_d(x, result.getData()[i], MPFR_RNDN);
	        mpfr_add(r, r, x, MPFR_RNDN);
	    }
	    mpfr_printf("result: %.70Rf \n", r);
	    /* printf("RAW Data:\n");
	    result.prettyPrint(); */
	    mpfr_clear(x);
	    mpfr_clear(r);
	}

//УМНОЖЕНИЕ СКАЛЯРА НА ВЕКТОР
template<int prec>
__global__ void campary_scal_kernel(multi_prec<prec> *alpha, multi_prec<prec> *x, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if( index < n){
        x[index] *= alpha[0];
    }
}
//ВЫЗОВ УМНОЖЕНИЯ СКАЛЯРА НА ВЕКТОР
template <int prec>
void campary_scal(int n, multi_prec<prec> *alpha, multi_prec<prec> *x){
    int BLOCKS = n / CAMPARY_VECTOR_MULTIPLY_THREADS + 1;
    campary_scal_kernel <prec> <<<BLOCKS, CAMPARY_VECTOR_MULTIPLY_THREADS>>>(alpha, x, n);
}


template<int prec>
void campary_scal_test(int n, mpfr_t alpha, mpfr_t *x, int convert_digits)
{


    //Host daa
    multi_prec<prec> halpha;
    multi_prec<prec> *hx = new multi_prec<prec>[n];

    //GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dx;

    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        hx[i] = convert_to_string_sci(x[i], convert_digits).c_str();
    }
    halpha = convert_to_string_sci(alpha, convert_digits).c_str();

    //Copying alpha to the GPU
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch

	cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
	campary_scal<prec>(n, dalpha, dx);


    //Copying to the host
    cudaMemcpy(hx, dx, sizeof(multi_prec<prec>) * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < n; i ++){
        hx[0] += hx[i];
    }
    printResult<prec>(hx[0]);

    //Cleanup
    delete [] hx;
    cudaFree(dalpha);
    cudaFree(dx);
}




template <typename Type>
Type vvod(Type c) //Шаблон функции проверки ввода числа
{
	for (;;)
	{
		if ((cin >> c))
		{
			std::cin.clear();
			std::cin.ignore(255, '\n');
			return c;
		}
		std::cout << "Неверно введеное число. Попробуйте ещё раз.\n";
		std::cin.clear();
		std::cin.ignore(255, '\n');
	}
}


int main()
{
int N =1024;
//УСТАНОВКА ТОЧНОСТИ
    INP_DIGITS = 10;
	 int count,*s=&count;
	 //int convert = 20;
	for (;;)
	{
		cout<<"Выберите операцию, 1-AXPY, 2-GEMV, 3-GEMM"<<endl;
		  switch(vvod(*s))
		  {
		  	  case 1:
		  	  {
		  		  //Инициализация
		  		    mpfr_t * vectorX;
		  		    mpfr_t * alpha;

		  		    vectorX = create_random_array(N, INP_BITS);
		  		    alpha = create_random_array(1, INP_BITS);
		  		    //Вызов

		  		 campary_scal_test<4>(N,alpha[0],vectorX,INP_DIGITS);

		  		for(int i = 0; i < N; i++){
		  		        mpfr_clear(vectorX[i]);
		  		    }
		  		    mpfr_clear(alpha[0]);


		  		    delete [] vectorX;
		  		    delete [] alpha;

		  		  break;

		  	  }
		  	  case 2:
		  	  {
		  		  break;
		  	  }
		  	  case 3:
		  	  {
		  		  break;
		  	  }
		  	default: cout<<"Выберите ещё раз."<<endl;
		  }

	}
	return 0;
}
