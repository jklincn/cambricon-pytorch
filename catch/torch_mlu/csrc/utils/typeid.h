#pragma once

template< class T>
struct on_chip_scalar_t {
  typedef T type;
  static constexpr size_t size = sizeof(type);
};

template<>
struct on_chip_scalar_t<double> {
  typedef float type;
  static constexpr size_t size = sizeof(type);
};

template<>
struct on_chip_scalar_t<int64_t> {
  typedef int type;
  static constexpr size_t size = sizeof(type);
};

template<>
struct on_chip_scalar_t<c10::complex<double> > {
  typedef c10::complex<float> type;
  static constexpr size_t size = sizeof(type);
};

template< class T>
typename on_chip_scalar_t<T>::type* cast_to_onchip_scalar_type(T *p){
    return reinterpret_cast<typename on_chip_scalar_t<T>::type*>(p);
};

template<class T>
struct is_complex {
  static constexpr bool value = false;
};

template<class T>
struct is_complex<c10::complex<T> > {
  static constexpr bool value = true;
};


template< class T>
struct decomplexed_scalar {
  typedef T type;
};


template< class T>
struct decomplexed_scalar<c10::complex<T> > {
  typedef T type;
};