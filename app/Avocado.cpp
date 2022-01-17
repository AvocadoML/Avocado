//============================================================================
// Name        : Avocado.cpp
// Author      : Maciej Kozarzewski
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/core/DataType.hpp>
#include <Avocado/math/tensor_operations.hpp>

using namespace avocado;
using namespace avocado::backend;

int main()
{
	std::cout << Device::hardwareInfo() << std::endl;

	Context context(Device::cuda(0));
	context.synchronize();

	Tensor t1(Shape( { 10, 10, 10 }), DataType::FLOAT32, Device::cpu());
	t1.setall(1.0f);
	t1.set(2.0f, { 1, 2, 3 });

	Tensor t2(Shape( { 10, 10, 10 }), DataType::FLOAT32, Device::cuda(0));

	math::copyTensor(context, t2, t1);
	context.synchronize();

	std::cout << t2.get<float>( { 1, 2, 3 }) << '\n';
	std::cout << t2.get<float>( { 1, 2, 2 }) << '\n';

	std::cout << "!!!Hello World!!!" << std::endl; // prints !!!Hello World!!!
	return 0;
}
