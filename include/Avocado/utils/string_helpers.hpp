/*
 * string_helpers.hpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_UTILS_STRING_HELPERS_HPP_
#define AVOCADO_UTILS_STRING_HELPERS_HPP_

#include <string>
#include <vector>

namespace avocado
{
	bool equals(const char *str1, const char *str2);
	int occurence(const std::string &str, char c);

	bool startsWith(const std::string &str, const std::string &seek);
	bool endsWith(const std::string &str, const std::string &seek);
	std::string trim(const std::string &str);

	std::vector<std::string> split(const std::string &str, char delimiter);
	bool isNumber(const std::string &str) noexcept;

	void print_string(const std::string &str);
	void println(const std::string &str);

} /* namespace avocado */

#endif /* AVOCADO_UTILS_STRING_HELPERS_HPP_ */
