#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <omp.h>
static inline float decode(uint16_t x){uint32_t u=uint32_t(x)<<16;float v;std::memcpy(&v,&u,4);return v;}
static inline uint16_t round_bf16(float v){uint32_t u;std::memcpy(&u,&v,4);return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);}
extern "C" int audit5(const uint16_t* original,const float* delta,uint64_t n,
 const float* coefficients,double* output,uint64_t* counts){
 int invalid=0;
 for(int j=0;j<17;j++)output[j]=0.;
 for(int j=0;j<16;j++)counts[j]=0;
 #pragma omp parallel num_threads(4)
 {
  double local[17]={0.};uint64_t nc[16]={0};int bad=0;
  #pragma omp for schedule(static)
  for(uint64_t i=0;i<n;i++){
   float w=decode(original[i]),d=delta[i];
   if(!std::isfinite(w)||!std::isfinite(d)){bad++;continue;}
   local[0]+=double(w)*double(w);local[1]+=double(d)*double(d);nc[0]+=(d==0.f);
   for(int j=0;j<5;j++){
    float scaled=d*coefficients[j];
    float combined=w+scaled;
    uint16_t bits=round_bf16(combined);float merged=decode(bits);
    if(!std::isfinite(merged)){bad++;continue;}
    double actual=double(merged)-double(w);bool changed=(actual!=0.f),bytechange=(bits!=original[i]);
    nc[1+3*j]+=changed;nc[2+3*j]+=bytechange;nc[3+3*j]+=(bytechange&&!changed);
    local[2+3*j]+=double(actual)*double(actual);
    local[3+3*j]+=double(actual)*double(d);
    local[4+3*j]=std::max(local[4+3*j],double(std::abs(actual)));
   }
  }
  #pragma omp critical
  {
   invalid+=bad;output[0]+=local[0];output[1]+=local[1];counts[0]+=nc[0];
   for(int j=0;j<5;j++){
    output[2+3*j]+=local[2+3*j];output[3+3*j]+=local[3+3*j];
    output[4+3*j]=std::max(output[4+3*j],local[4+3*j]);
    for(int k=0;k<3;k++)counts[1+3*j+k]+=nc[1+3*j+k];
   }
  }
 }
 return invalid;
}
extern "C" int merge1(const uint16_t* original,const float* delta,uint64_t n,
 float coefficient,uint16_t* target,double* stats,uint64_t* counts){
 int invalid=0;double base_ss=0.,delta_ss=0.,max_delta=0.;uint64_t changed=0,bytechanged=0;
 #pragma omp parallel for num_threads(4) reduction(+:invalid,base_ss,delta_ss,changed,bytechanged) reduction(max:max_delta)
 for(uint64_t i=0;i<n;i++){
  float w=decode(original[i]),d=delta[i];
  if(!std::isfinite(w)||!std::isfinite(d)){invalid++;continue;}
  float scaled=d*coefficient;float combined=w+scaled;
  uint16_t bits=round_bf16(combined);float merged=decode(bits);
  if(!std::isfinite(merged)){invalid++;continue;}
  target[i]=bits;double diff=double(merged)-double(w);
  changed+=(diff!=0.f);bytechanged+=(bits!=original[i]);
  base_ss+=double(w)*double(w);delta_ss+=double(diff)*double(diff);
  max_delta=std::max(max_delta,double(std::abs(diff)));
 }
 stats[0]=base_ss;stats[1]=delta_ss;stats[2]=max_delta;counts[0]=changed;counts[1]=bytechanged;
 return invalid;
}
