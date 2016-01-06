//////////////////////////////////////////////////////////////////////////
/** 
*   Simple Math Library - Basic and common mathematical computations 
*
*   Code written by Shuo Jin at Dept. of MAE, CUHK, Hong Kong
*   Contact: jerry.shuojin@gmail.com. All Rights Reserved.
*/
//////////////////////////////////////////////////////////////////////////

#ifndef HEADER_SMLLIB_H
#define HEADER_SMLLIB_H

#include <cmath>

namespace sml
{
    /** Return the maximum value of a and b */
	#ifdef max
	#undef max
	#endif
	template<typename T>
	const T max(const T _a, const T _b)
	{
		return _a > _b ? _a : _b;
	}

	/** Return the minimum value of a and b */
	#ifdef min
	#undef min
	#endif
	template<typename T>
	const T min(const T _a, const T _b)
	{
		return _a < _b ? _a : _b;
	}

	/** Return the sum of the elements of a vector */
	template<typename T>
	const T sum(T* _v, const size_t _k)
	{
		T re(0);
		for (size_t i = 0; i < _k; ++i)
		{
			re += _v[i];
		}
		return re;
	}

	/** Return the L0 norm of a vector */
	template<typename T>
	const size_t L0_norm(T* _v, const size_t _k)
	{
		size_t norm(0);
		for (size_t i = 0; i < _k; ++i)
		{
			if (_v[i]) ++norm;
		}
		return norm;
	}

	/** Return the L1 norm of a vector */
	template<typename T>
	const T L1_norm(T* _v, const size_t _k)
	{
		T norm(0);
		for (size_t i = 0; i < _k; ++i)
		{
			norm += abs(_v[i]);
		}
		return norm;
	}

	/** Return the L2 norm of a vector
	*   \T1 is the data type of the input value
	*   \T2 is the data type of the return value
	*/
	template<typename T1, typename T2 = double>
	const T2 L2_norm(T1* _vec, const size_t _k)
	{
		T2 norm(0);
		for (size_t i = 0; i < _k; ++i)
		{
			norm += _vec[i] * _vec[i];
		}
		return sqrt(norm);
	}

	/** Return the L_infi norm of a vector */
	template<typename T>
	const T L_infi_norm(T* _v, const size_t _k)
	{
		T norm(0);
		for (size_t i = 0; i < _k; ++i)
		{
			norm  = abs(_v[i]) > norm ? abs(_v[i]) : norm;
		}
		return norm;
	}

	/** Return the Euclidean distance of two k-dimensional vectors
	*   \T1 is the data type of the input value
	*   \T2 is the data type of the return value
	*/
	template<typename T1, typename T2 = double>
	const T2 dist(T1* _vec1, T1* _vec2, const size_t _k)
	{
		T2 dist(0);
		for (size_t i = 0; i < _k; ++i)
		{
			dist += (_vec1[i] - _vec2[i]) * (_vec1[i] - _vec2[i]);
		}
		return sqrt(dist);
	}

	/** Return the squared Euclidean distance of two k-dimensional vectors
	*   \T1 is the data type of the input value
	*   \T2 is the data type of the return value
	*/
	template<typename T1, typename T2 = double>
	const T2 sqr_dist(T1* _vec1, T1* _vec2, const size_t _k)
	{
		T2 sqr_dist(0);
		for (size_t i = 0; i < _k; ++i)
		{
			sqr_dist += (_vec1[i] - _vec2[i]) * (_vec1[i] - _vec2[i]);
		}
		return sqr_dist;
	}

	/** Return the cross product of two three-dimensional vectors
	*   \T should be float/double data type, not compatible with int/long etc.
	*/
	template<typename T = double>
	void cross(T _v1[3], T _v2[3], T _output[3])
	{
		_output[0] = _v1[1] * _v2[2] - _v1[2] * _v2[1];
		_output[1] = _v1[2] * _v2[0] - _v1[0] * _v2[2];
		_output[2] = _v1[0] * _v2[1] - _v1[1] * _v2[0];
	}

	/** Return the dot product of two three dimensional vectors
	*   \T should be float/double data type, not compatible with int/long etc.
	*/
	template<typename T = double>
	const T dot(T _v1[3], T _v2[3])
	{
		return _v1[0] * _v2[0] + _v1[1] * _v2[1] + _v1[2] * _v2[2];
	}

	/** Return the normalized vector of the input
	*   \T should be float/double data type, not compatible with int/long etc.
	*   Return true if the operation of normalization is successful
	*/
	template<typename T = double>
	const bool normalize(T* _v, const size_t _k)
	{
		T sqr_norm(0);
		for (size_t i = 0; i < _k; ++i)
		{
			sqr_norm += _v[i] * _v[i];
		}

		if (sqr_norm == 0) return false;

		sqr_norm = sqrt(sqr_norm);
		for (size_t i = 0; i < _k; ++i)
		{
			_v[i] /= sqr_norm;
		}

		return true;
	}

	/** Return the integration result of a linear function ax+b from x1 to x2 (x1 < x2)
	*   \T should be float/double data type, not compatible with int/long etc.
	*/
	template<typename T = double>
	const T linear_integration(const T _a, const T _b, const T _x1, const T _x2)
	{	
		T val1 = 0.5 * _a * _x1 * _x1 + _b * _x1;
		T val2 = 0.5 * _a * _x2 * _x2 + _b * _x2;
		return val2 - val1;
	}

	/** Return the integration result of a non-negative linear function |ax+b| from x1 to x2 (x1 < x2)
	*   \T should be float/double data type, not compatible with int/long etc.
	*/
	template<typename T = double>
	const T linear_absolute_integration(const T _a, const T _b, const T _x1, const T _x2)
	{
		if ((_a * _x1 + _b) * (_a * _x2 + _b) >= 0.0)
		{
			T val1 = 0.5 * _a * _x1 * _x1 + _b * _x1;
			T val2 = 0.5 * _a * _x2 * _x2 + _b * _x2;
			return abs(val2 - val1);
		}
		else
		{
			T h1 = abs(_a * _x1 + _b);
			T h2 = abs(_a * _x2 + _b);
			T t = _x1 + h1 / (h1 + h2) * (_x2 - _x1);

			T val1 = 0.5 * _a * _x1 * _x1 + _b * _x1;
			T valt = 0.5 * _a * t * t + _b * t;
			T val2 = 0.5 * _a * _x2 * _x2 + _b * _x2;

			return abs(valt - val1) + abs(val2 - valt);
		}
	}

	/** Search a vector to check if a value is in the vector by brute force
	*   Return the position index of the matched value if found
	*   Return the dimension _sz of the vector if not found
	*/
	template<typename T>
	const size_t bruteforce_search(T* _vec, const size_t _sz, const T _query)
	{
		size_t pos;
		for (pos = 0; pos < _sz; ++pos)
		{
			if (_query == _vec[pos])
			{
				return pos;
			}
		}
		return pos;
	}

} // namespace sml

#endif // HEADER_SMLLIB_H