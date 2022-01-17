//============================================================================
// Name        : Avocado.cpp
// Author      : Maciej Kozarzewski
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/DataType.hpp>
#include <Avocado/backend/backend_libraries.hpp>
#include <ReferenceBackend/reference_backend.h>

using namespace avocado;
using namespace avocado::backend;

int main()
{
	avMemoryDescriptor_t mem;
	cpuCreateMemoryDescriptor(&mem, 4000);
	avContextDescriptor_t context = cpuGetDefaultContext();
//	avStatus_t status = cpuSetMemory(context, mem, 0, 4000, nullptr, 0);
//	std::cout << status << '\n';
	cpuDestroyMemoryDescriptor(mem);

//	cudaCreateMemoryDescriptor(&mem, 0, 4000);
//	avContextDescriptor_t context = cudaGetDefaultContext(1);
//	avStatus_t status = cudaSetMemory(context, mem, 0, 4000, nullptr, 0);
//	std::cout << status << '\n';
//	cudaDestroyMemoryDescriptor(mem);

//	Tensor t(Shape( { 10, 10, 10 }), DataType::FLOAT32, Device::cpu());
//	t.set(1.0f, { 1, 2, 3 });

//	std::cout << t.get<float>( { 1, 2, 3 }) << '\n';

	std::cout << "!!!Hello World!!!" << std::endl; // prints !!!Hello World!!!
	return 0;
}
