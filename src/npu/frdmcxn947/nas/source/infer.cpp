#include <stdio.h>
#include <unistd.h>
#include "fsl_debug_console.h"
#include "model.h"
#include "timer.h"
#include "math.h"

#define MODEL_IN_W 32
#define MODEL_IN_H  32
#define MODEL_IN_C 3
#define N_CLASSES 10
#define MODEL_IN_COLOR_BGR 0

#define Q15_MAX_VALUE   32767
#define Q15_MIN_VALUE   -32768

typedef int32_t q31_t;
typedef int16_t q15_t;

__attribute__((section (".model_input_buffer"))) static uint8_t model_input_buf[MODEL_IN_W*MODEL_IN_H*MODEL_IN_C] = {0};

static float processed_output[N_CLASSES];

void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out)
{
    q31_t     sum;
    int16_t   i;
    uint8_t   shift;
    q31_t     base;
    base = -1 * 0x80000000;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }

    base = base - (16<<14);

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            shift = (uint8_t)((8192 + vec_in[i] - base) >> 14);
            sum += (0x1 << shift);
        }
    }


    /* This is effectively (0x1 << 32) / sum */
    int64_t div_base = 0x100000000LL;
    int32_t output_base = (int32_t)(div_base / sum);
    int32_t out;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            shift = (uint8_t)(17+((8191 + base - vec_in[i]) >> 14));

            out = (output_base >> shift);

            if (out > 32767)
            	out = 32767;

            p_out[i] = (q15_t)out;


        } else
        {
            p_out[i] = 0;
        }
    }

}

q31_t** generateArray() {
    q31_t** array = (q31_t**)malloc(sizeof(q31_t*) * 1);
    array[0] = (q31_t*)malloc(sizeof(q31_t) * N_CLASSES);  

    q31_t value = 0;
    for (int i = 0; i < N_CLASSES; i++) {
        array[0][i] = value++; 
    }

    return array;
}

void delay_ms(uint32_t ms) {
    uint32_t start = TIMER_GetTimeInUS();
    while ((TIMER_GetTimeInUS() - start) < (ms * 1000)) {
        // Busy-wait
    }
}

extern "C" {

	uint32_t s_Us = 0;
	volatile int8_t g_isImgBufReady = 0;
	#define WND_X0 0
	#define WND_Y0 0

	void run()
	{
		tensor_dims_t inputDims;
		tensor_type_t inputType;
		uint8_t* inputData;

		tensor_dims_t outputDims;
		tensor_type_t outputType;
		uint8_t* outputData;
		size_t arenaSize;

		delay_ms(5000);

		auto startTime = TIMER_GetTimeInUS();
		if (MODEL_Init() != kStatus_Success)
		{
			PRINTF("Failed initializing model");
			for (;;) {}
		}
		auto endTime = TIMER_GetTimeInUS();
		auto dt = endTime - startTime;
		s_Us = (uint32_t)dt;
		PRINTF("Init: %d us\n", s_Us);

		size_t usedSize = MODEL_GetArenaUsedBytes(&arenaSize);
		PRINTF("\r\n%d/%d kB (%0.2f%%) tensor arena used\r\n", usedSize / 1024, arenaSize / 1024, 100.0*usedSize/arenaSize);

		inputData = MODEL_GetInputTensorData(&inputDims, &inputType);
		outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);

		TfLiteTensor* outputTensor = MODEL_GetOutputTensor(0);

		delay_ms(5000);
		
		while(1)
		{
			startTime = TIMER_GetTimeInUS();
			uint8_t *pOut = inputData;
			for (int i = 0; i < MODEL_IN_W; i++) {
				for (int j = 0; j < MODEL_IN_H; j++) {
					for (int c = 0; c < MODEL_IN_C; c++) {
						*pOut++ = 0;
					}
				}
			}
			uint8_t *buf = 0;
			memset(inputData,0,inputDims.data[1]*inputDims.data[2]*inputDims.data[3]);
			buf = inputData + (inputData,inputDims.data[1] - MODEL_IN_H) /2 * MODEL_IN_W * MODEL_IN_C;
			memcpy(buf, model_input_buf, MODEL_IN_W*MODEL_IN_H*MODEL_IN_C);
			endTime = TIMER_GetTimeInUS();
			dt = endTime - startTime;
			s_Us = (uint32_t)dt;
			PRINTF("Memory I/O: %d us\n", s_Us);

			delay_ms(5000);

			startTime = TIMER_GetTimeInUS();
			MODEL_RunInference();
			endTime = TIMER_GetTimeInUS();
			dt = endTime - startTime;
			s_Us = (uint32_t)dt;
			PRINTF("Inference: %d us\n", s_Us);

			delay_ms(5000);

			size_t output_size = N_CLASSES * sizeof(int8_t);
			startTime = TIMER_GetTimeInUS();
			memcpy(processed_output, outputData, output_size);
			endTime = TIMER_GetTimeInUS();
			s_Us = (uint32_t)dt;
			PRINTF("start_time: %d \n", (uint32_t)startTime);
			PRINTF("end_time: %d \n", (uint32_t)endTime);
			PRINTF("Memory I/0: %d us\n", s_Us);

			delay_ms(5000);

			q31_t** array = generateArray();
			q15_t p_out[N_CLASSES];
			startTime = TIMER_GetTimeInUS();
			softmax_q17p14_q15(array[0], N_CLASSES, p_out);
			endTime = TIMER_GetTimeInUS();
			dt = endTime - startTime;
			s_Us = (uint32_t)dt;
			PRINTF("start_time: %d \n", (uint32_t)startTime);
			PRINTF("end_time: %d \n", (uint32_t)endTime);
			PRINTF("Post-processing: %d us\n", s_Us);
		}

	}
}


