//////////////////////////////////////////////////////////////////////////
/**
*   C++ implementation of Mean Shift Clustering Method
*
*   This implementation exploits ANN search to speed up its performance
*   ANN library should be configured properly for this feature.
*
*   Gaussian kernel is used in this implementation.
*   
*   Current version uses a simple adaptive bandwidth, which is not optimal.
*   The distance of the 0.05*vectornum-th point to a point is used as the bandwidth.
*   Reference [2] provides a complex adaptive strategy to compute bandwidth.
*
*   Reference:
*   1. Mean shift: a robust approach toward feature space analysis
*   Link: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1000236&tag=1
*   2. The variable bandwidth mean shift and data-driven scale selection
*   Link: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=937550
*   3. A blog introducing mean shift
*   Link: http://bit.ly/1NSW5km
*
*	Version 0.20151216, not optimized.
*
*   Code developed by Shuo Jin at Dept. of MAE, CUHK, Hong Kong
*   Email: jerry.shuojin@gmail.com. All rights reserved.
*/
//////////////////////////////////////////////////////////////////////////

#ifndef HEADER_MEAN_SHIFT_H
#define HEADER_MEAN_SHIFT_H

#include "sml.h"

#include <vector>
#include <limits>
#include <iostream>
#include <ctime>

/** Configure the path of ANN library here */
#include "..\ANN\ANN.h"

/** Some information will be output on console if this macro is defined
*   Uncomment this macro to disable this feature
*/
#define __MEAN_SHIFT_ENABLE_CONSOLE_OUTPUT__

/** It will use KNN to search K neighbors in computing mean shift vector if this macro is defined 
*   Uncomment this macro to disable this feature
*/
#define __MEAN_SHIFT_ENABLE_KNN_SEARCH__

/** Class ms_vec defines a D-dimensional vector
*   \T Data type of vector
*   \D Dimension of vector
*/
template <typename T, size_t D>
class ms_vec
{
public:	
	/** Default constructor*/
	ms_vec()
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] = std::numeric_limits<T>::lowest();
	}
	
	/** Create a vector with a initial value _v for all elements */
	ms_vec(const T _v)
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] = _v;
	}

	/** Copy constructor*/
	ms_vec(const ms_vec<T, D> & _vec)
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] = _vec.msv_data[i];
	}

	/** Destructor*/
	~ms_vec()
	{

	}

public:
	/** Reset the vector with a specified value _v */
	void reset(const T _v = 0)
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] = _v;
	}

	/** Operator [] */
	T & operator [] (const size_t _i)
	{
		return msv_data[_i];
	}

	const T & operator [] (const size_t _i) const
	{
		return msv_data[_i];
	}

	/** Operator = */
	ms_vec<T, D> & operator = (const ms_vec<T, D> & _vec)
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] = _vec.msv_data[i];
		return *this;
	}

	/** Operator += */
	ms_vec<T, D> & operator += (const ms_vec<T, D> & _vec)
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] += _vec[i];
		return *this;
	}

	/** Operator -= */
	ms_vec<T, D> & operator -= (const ms_vec<T, D> & _vec)
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] -= _vec[i];
		return *this;
	}

	/** Operator *= */
	ms_vec<T, D> & operator *= (const T _v)
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] *= _v;
		return *this;
	}

	/** Operator /= */
	ms_vec<T, D> & operator /= (const T _v)
	{
		for (size_t i = 0; i < D; ++i) msv_data[i] /= _v;
		return *this;
	}

	/** get the raw data vector */
	T* raw_data()
	{
		return msv_data;
	}

	/** ************* Friend functions to provide more operators ****************************** */
	/** Operator + */
	friend ms_vec<T, D> operator + (const ms_vec<T, D> & _vec1, const ms_vec<T, D> & _vec2)
	{
		return ms_vec<T, D>(_vec1) += _vec2;
	}

	/** Operator - */
	friend ms_vec<T, D> operator - (const ms_vec<T, D> & _vec1, const ms_vec<T, D> & _vec2)
	{
		return ms_vec<T, D>(_vec1) -= _vec2;
	}

	/** Operator * */
	friend ms_vec<T, D> operator * (const T _v, const ms_vec<T, D> & _vec)
	{
		return ms_vec<T, D>(_vec) *= _v;
	}

	friend ms_vec<T, D> operator * (const ms_vec<T, D> & _vec, const T _v)
	{
		return ms_vec<T, D>(_vec) *= _v;
	}

	/** Operator / */
	friend ms_vec<T, D> operator / (const ms_vec<T, D> & _vec, const T _v)
	{
		return ms_vec<T, D>(_vec) /= _v;
	}

	/** Operator == */
	friend const bool operator == (const ms_vec<T, D> & _vec1, const ms_vec<T, D> & _vec2)
	{
		for (size_t i = 0; i < D; ++i)
		{
			if (!_vec1[i] == _vec2[i]) return false;
		}
		return true;
	}

	/** Operator != */
	friend const bool operator != (const ms_vec<T, D> & _vec1, const ms_vec<T, D> & _vec2)
	{
		for (size_t i = 0; i < D; ++i)
		{
			if (!_vec1[i] == _vec2[i]) return true;
		}
		return false;
	}

	/** Operator << */
	friend std::ostream & operator << (std::ostream & _os, const ms_vec<T, D> & _vec)
	{
		_os << "[";
		for (size_t i = 0; i < D - 1; ++i) _os << _vec[i] << " ";
		_os << _vec[D - 1] << "]";
		return _os;
	}

private:
	T msv_data[D];
};

/** ************************************************************************************* */

/** Class ms_vec_clstr defines a cluster of D-dimensional vectors
*   The center of cluster is a mode of mean shift
*   \T Data type of vector
*   \D Dimension of vector
*/
template <typename T, size_t D>
class ms_vec_clstr
{
public:
	/** Default constructor*/
	ms_vec_clstr() : msvc_mode_center(0)
	{

	}

	/** Destructor*/
	~ms_vec_clstr()
	{

	}
	
public:
	/** Add a vector to this cluster with its mean shift result
	*   \_vec    The vector to be added to this cluster
	*   \_ms_re  The mean shift result of this vector
	*/
	void add_vec(const ms_vec<T, D> _vec, const ms_vec<T, D> _ms_re)
	{
		T sz = static_cast<T> (msvc_vec_collection.size());
		msvc_mode_center = (msvc_mode_center * sz + _vec) / (sz + 1);

		msvc_vec_collection.push_back(_vec);
	}

	/** Get the current center of this cluster */
	const ms_vec<T, D> mode_center() const
	{
		return msvc_mode_center;
	}

	/** Compute the distance of a vector to this center */
	const T dist_to_center(ms_vec<T, D> _vec)
	{
		return sml::dist<T, T>(msvc_mode_center.raw_data(), _vec.raw_data(), D);
	}

	/** Get the size of this cluster*/
	const size_t size() const
	{
		return msvc_vec_collection.size();
	}

	/** Clear this cluster */
	void clear()
	{
		msvc_vec_collection.clear();
		msvc_mode_center.reset();
	}

	/** Check if this cluster is empty or not */
	const bool empty() const
	{
		return !(msvc_vec_collection.size());
	}

private:
	std::vector< ms_vec<T, D> > msvc_vec_collection;

	ms_vec<T, D> msvc_mode_center;
};

/** ************************************************************************************* */

/** Class mean_shift defines the mean shift method
*   The current version uses a simple strategy
*   \T Data type of vector
*   \D Dimension of vector
*/
template <typename T, size_t D>
class mean_shift
{
public:
	/** Default constructor*/
	mean_shift() : 
		ms_kdtree(nullptr),
		ms_epsilon(0.001),
		ms_mode_bound(1),
		ms_max_iter_size(200),
		ms_nn_num(100)
	{

	}

	/** Destructor*/
	~mean_shift()
	{
		clear();
	}

public:
	/** Add a vector to this class */
	void add_vec(ms_vec<T, D> _vec)
	{
		ms_data_vecs.push_back(_vec);
	}

	/** Get the number of clusters after doing mean shift */
	const size_t clstr_count() const
	{
		return ms_vec_clstrs.size();
	}

	/** Get a generated cluster */
	ms_vec_clstr<T, D> & clstr(const size_t _i)
	{
		return ms_vec_clstrs[_i];
	}

	/** Get the belonged cluster index of a vector */
	const size_t clstr_idx_of_vec(const size_t _i)
	{
		return ms_vec_clstr_idx[_i];
	}

	/** Set the condition to check termination
	*   This value is used to determine the end of a mean shift process for a vector. 
	*/
	void set_epsilon(const T _e)
	{
		ms_epsilon = _e;
	}

	/** Set the condition to check mode belonging
	*	It is used to judge if a mean shift result belongs to an existing mode or a new mode should be created.
	*   The default value is set to 1
	*/
	void set_mode_bound(const T _mb)
	{
		ms_mode_bound = _mb;
	}

	/** Set the maximum step size for finding the mode of a vector iteratively
	*   The default value is set to 200
	*/
	void set_max_iter_size(const size_t _s)
	{
		ms_max_iter_size = _s;
	}

	/** Set the number of nearest neighbors for KNN search to compute mean shift
	*   The default value is set to 100
	*/
	void set_knn_search_num(const size_t _k)
	{
		ms_nn_num = _k;
	}

	/** Do mean shift computation to find all modes of the input vectors */
	void find_modes();

	/** Clear all data */
	void clear()
	{
		ms_data_vecs.clear();
		ms_vec_clstrs.clear();
		ms_data_bw.clear();
		ms_data_bw_pow.clear();
		ms_vec_clstr_idx.clear();
	}

	/** Output mean shift result on console */
	void output_on_console()
	{
		for (size_t i = 0; i < ms_vec_clstrs.size(); ++i)
		{
			std::cout << ms_vec_clstrs[i].mode_center() << std::endl;
		}

		std::cout << "MS CLSTR COUNT = " << ms_vec_clstrs.size() << std::endl;
	}

private:
	/** Member variables */
	std::vector< ms_vec<T, D> > ms_data_vecs;

	std::vector< T > ms_data_bw; 

	std::vector< T > ms_data_bw_pow;

	std::vector< ms_vec_clstr<T, D> > ms_vec_clstrs;

	std::vector< size_t >  ms_vec_clstr_idx;

	ANNkd_tree* ms_kdtree;

	T ms_search_radius;

	T ms_epsilon;

	T ms_mode_bound;

	size_t ms_max_iter_size;

	size_t ms_nn_num;

	/** Auxilliary functions */
	void ms_init_kdtree();

	void ms_destory_kdtree();

	void ms_init_bandwidth_radius();

	const ms_vec<T, D> ms_find_vec_mode(ms_vec<T, D>);

	const size_t ms_find_match_clstr(ms_vec<T, D>);
};

template <typename T, size_t D>
void mean_shift<T, D>::find_modes()
{
#ifdef __MEAN_SHIFT_ENABLE_CONSOLE_OUTPUT__

	std::clock_t begin = std::clock();

	printf("Init...\n");

#endif

	ms_data_vecs.shrink_to_fit();

	ms_vec_clstr_idx.resize(ms_data_vecs.size(), std::numeric_limits<size_t>::infinity());

	ms_init_kdtree();

	ms_init_bandwidth_radius();

#ifdef __MEAN_SHIFT_ENABLE_CONSOLE_OUTPUT__

	printf("Mean Shift Progress: 00%%");

#endif

	for (size_t i = 0; i < ms_data_vecs.size(); ++i)
	{
		ms_vec<T, D> result = ms_find_vec_mode(ms_data_vecs[i]);

		const size_t match_id = ms_find_match_clstr(result);

		if (match_id == ms_vec_clstrs.size())
		{
			ms_vec_clstrs.push_back(ms_vec_clstr<T, D>());			
		}

		ms_vec_clstrs[match_id].add_vec(ms_data_vecs[i], result);
		ms_vec_clstr_idx[i] = match_id;

#ifdef __MEAN_SHIFT_ENABLE_CONSOLE_OUTPUT__

		printf("\b\b\b%2d%%", static_cast<int>((i + 1) * 100 / double(ms_data_vecs.size())));

#endif

	}

#ifdef __MEAN_SHIFT_ENABLE_CONSOLE_OUTPUT__
	
	std::clock_t end = std::clock();
	
	printf("\n");
	printf("Total time: %.3lf secs\n", (end - begin) / double(CLOCKS_PER_SEC));

#endif

	ms_destory_kdtree();

	ms_vec_clstrs.shrink_to_fit();
}

template <typename T, size_t D>
void mean_shift<T, D>::ms_init_kdtree()
{
	ANNpointArray ann_points = annAllocPts(ms_data_vecs.size(), D);

	for (size_t i = 0; i < ms_data_vecs.size(); ++i)
	{
		for (size_t j = 0; j < D; ++j)
		{
			ann_points[i][j] = ms_data_vecs[i][j];
		}
	}

	ms_kdtree = new ANNkd_tree(ann_points, ms_data_vecs.size(), D);
}

template <typename T, size_t D>
void mean_shift<T, D>::ms_destory_kdtree()
{
	delete ms_kdtree;
	ms_kdtree = nullptr;

	annClose();
}

template <typename T, size_t D>
void mean_shift<T, D>::ms_init_bandwidth_radius()
{
	/** The number of neighbors to determine bandwidth */
	const size_t k = static_cast<size_t> (0.05 * ms_data_vecs.size());

	/** The magnification factor to determine search radius */
	const size_t m = 1;

	ms_data_bw.resize(ms_data_vecs.size(), 0);
	ms_data_bw_pow.resize(ms_data_vecs.size(), 1);

	ANNpoint pt = annAllocPt(D);
	ANNidxArray nkidx = new ANNidx[k];
	ANNdistArray nkdist = new ANNdist[k];

	for (size_t pt_it = 0; pt_it < ms_data_vecs.size(); ++pt_it)
	{
		for (size_t i = 0; i < D; ++i) pt[i] = ms_data_vecs[pt_it][i];

		ms_kdtree->annkSearch(pt, k, nkidx, nkdist);

		ms_data_bw[pt_it] = static_cast<T> (sqrt(nkdist[k - 1]));

		for (size_t i = 0; i < D + 2; ++i) ms_data_bw_pow[pt_it] *= ms_data_bw[pt_it];
	}

	annDeallocPt(pt);
	delete[] nkidx;
	delete[] nkdist;

#ifdef __MEAN_SHIFT_ENABLE_KNN_SEARCH__

	ms_search_radius = std::numeric_limits<T>::lowest();
	for (size_t i = 0; i < ms_data_bw.size(); ++i)
	{
		T temp_radius = m * ms_data_bw[i];
		ms_search_radius = temp_radius > ms_search_radius ? temp_radius : ms_search_radius;
	}

#endif

	//for (size_t pt_it = 0; pt_it < ms_data_vecs.size(); ++pt_it)
	//{
	//	ms_data_bw[pt_it] = 1;

	//	ms_data_bw_pow[pt_it] = 1;
	//}

	//ms_search_radius = 5;
}

template <typename T, size_t D>
const ms_vec<T, D> mean_shift<T, D>::ms_find_vec_mode(ms_vec<T, D> _vec)
{
	/** The number of neighbors to compute mean shift */	
	ms_vec<T, D> vec1(_vec), vec2(0);

	ANNpoint pt = annAllocPt(D);
	ANNdist sqr_rad = ms_search_radius * ms_search_radius;

	for (size_t ms_it = 0; ms_it < ms_max_iter_size; ++ms_it)
	{

#ifdef __MEAN_SHIFT_ENABLE_KNN_SEARCH__

		/** Use KNN search to speed up | Comment this block if necessary */
		for (size_t i = 0; i < D; ++i) pt[i] = vec1[i];

		ANNidxArray nnidx = new ANNidx[ms_nn_num];
		ANNdistArray nndist = new ANNdist[ms_nn_num];

		ms_kdtree->annkSearch(pt, ms_nn_num, nnidx, nndist);

		T denominator = 0;
		for (size_t i = 0; i < ms_nn_num; ++i)
		{
			T w = exp(-0.5 * nndist[i] / (ms_data_bw[nnidx[i]] * ms_data_bw[nnidx[i]])) / ms_data_bw_pow[nnidx[i]];

			denominator += w;

			vec2 += w * ms_data_vecs[nnidx[i]];
		}
		vec2 /= denominator;

		delete[] nnidx;
		delete[] nndist;

#else

		/** Traverse all data points in each step | Comment this block if necessary */
		T denominator = 0;
		for (size_t i = 0; i < ms_data_vecs.size(); ++i)
		{
			T dist = sml::dist<T, T>(vec1.raw_data(), ms_data_vecs[i].raw_data(), 2);

			T w = exp(-0.5 * dist * dist / (ms_data_bw[i] * ms_data_bw[i])) / ms_data_bw_pow[i];

			denominator += w;

			vec2 += w * ms_data_vecs[i];
		}
		vec2 /= denominator;

#endif

		if (sml::L2_norm<T, T>((vec2 - vec1).raw_data(), D) < ms_epsilon)
		{			
			return vec2;
		}

		vec1 = vec2;
		vec2.reset(0);
	}

#ifdef __MEAN_SHIFT_ENABLE_CONSOLE_OUTPUT__

	printf("MEAN SHIFT: Exceed maximum iteration step size.\n");

#endif	

	return vec1;
}

template <typename T, size_t D>
const size_t mean_shift<T, D>::ms_find_match_clstr(ms_vec<T, D> _vec)
{
	size_t clstr_it = 0;
	
	for (; clstr_it < ms_vec_clstrs.size(); ++clstr_it)
	{
		if (ms_vec_clstrs[clstr_it].dist_to_center(_vec) <= ms_mode_bound)
		{
			break;
		}
	}

	return clstr_it;
}



/** ************************************************************************************* */

/** Common definitions */
typedef ms_vec<double, 2>       ms_vec_d2;
typedef ms_vec_clstr<double, 2> ms_vec_clstr_d2;
typedef mean_shift<double, 2>   mean_shift_d2;

typedef ms_vec<float, 2>        ms_vec_f2;
typedef ms_vec_clstr<float, 2>  ms_vec_clstr_f2;
typedef mean_shift<float, 2>    mean_shift_f2;

typedef ms_vec<double, 3>       ms_vec_d3;
typedef ms_vec_clstr<double, 3> ms_vec_clstr_d3;
typedef mean_shift<double, 3>   mean_shift_d3;

typedef ms_vec<float, 3>        ms_vec_f3;
typedef ms_vec_clstr<float, 3>  ms_vec_clstr_f3;
typedef mean_shift<float, 3>    mean_shift_f3;

#endif // HEADER_MEAN_SHIFT_H