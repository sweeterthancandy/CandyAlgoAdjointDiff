sweeterthancady => candy algorithmic differentiation
====================================================

The idea of this is to have an in-memory expression tree which can be used to generate C++/C code

Scope is to be double only, with control flow statements

```c++
struct BlackScholesCallOption{
        template<class Double>
        struct Build{
                Double Evaluate(
                        Double t,
                        Double T,
                        Double r,
                        Double S,
                        Double K,
                        Double vol )const
                {
                        using MathFunctions::Phi;
                        using MathFunctions::Exp;
                        using MathFunctions::Pow;
                        using MathFunctions::Log;

                        Double d1 = ((1.0 / ( vol * Pow((T - t),0.5) )) * ( Log(S / K) +   (r + ( Pow(vol,2.0) ) / 2 ) * (T - t) ));
                        Double d2 = d1 - vol * (T - t);
                        Double pv = K * Exp( -r * ( T - t ) );
                        Double black = Phi(d1) * S - Phi(d2) * pv;
                        return black;
                }
        };
};
```


From this, I can both evaulate in code, or with an embedded dsl

We can generate both forward and backward code

![forward/backward eval graph](https://github.com/sweeterthancandy/sweeterthancady/blob/master/GithubAssets/graph.png)

```c++
double black_backwards(double t, double* d_t, double T, double* d_T, double r, double* d_r, double S, double* d_S, double K, double* d_K, double vol, double* d_vol){
    double const w2 = vol;
    double const w3 = T;
    double const w4 = t;
    double const w5 = ((w3)-(w4));
    double const w7 = std::pow(w5, 0.5);
    double const w8 = ((w2)*(w7));
    double const w9 = ((1)/(w8));
    double const w10 = S;
    double const w11 = K;
    double const w12 = ((w10)/(w11));
    double const w13 = std::log(w12);
    double const w14 = r;
    double const w16 = std::pow(w2, 2);
    double const w17 = ((w16)/(2));
    double const w18 = ((w14)+(w17));
    double const w19 = ((w18)*(w5));
    double const w20 = ((w13)+(w19));
    double const w21 = ((w9)*(w20));
    double const w22 = std::erfc(-(w21)/std::sqrt(2))/2;
    double const w23 = ((w22)*(w10));
    double const w24 = ((w2)*(w5));
    double const w25 = ((w21)-(w24));
    double const w26 = std::erfc(-(w25)/std::sqrt(2))/2;
    double const w27 = (-(w14));
    double const w28 = ((w27)*(w5));
    double const w29 = std::exp(w28);
    double const w30 = ((w11)*(w29));
    double const w31 = ((w26)*(w30));
    double const value = ((w23)-(w31));
    double const __rev_ad_value = 1;
    double const w33 = __rev_ad_value;
    double const __rev_ad_w31 = ((-1)*(w33));
    double const w34 = __rev_ad_w31;
    double const __rev_ad_w30 = ((w26)*(w34));
    double const w35 = __rev_ad_w30;
    double const __rev_ad_w29 = ((w11)*(w35));
    double const w36 = __rev_ad_w29;
    double const __rev_ad_w28 = ((w29)*(w36));
    double const w37 = __rev_ad_w28;
    double const __rev_ad_w27 = ((w5)*(w37));
    double const __rev_ad_w26 = ((w30)*(w34));
    double const w38 = std::pow(w25, 2);
    double const w39 = ((0.5)*(w38));
    double const w40 = (-(w39));
    double const w41 = std::exp(w40);
    double const w43 = ((w41)/(2.50663));
    double const w44 = __rev_ad_w26;
    double const __rev_ad_w25 = ((w43)*(w44));
    double const w45 = __rev_ad_w25;
    double const __rev_ad_w24 = ((-1)*(w45));
    double const __rev_ad_w23 = __rev_ad_value;
    double const w46 = __rev_ad_w23;
    double const __rev_ad_w22 = ((w10)*(w46));
    double const w47 = std::pow(w21, 2);
    double const w48 = ((0.5)*(w47));
    double const w49 = (-(w48));
    double const w50 = std::exp(w49);
    double const w51 = ((w50)/(2.50663));
    double const w52 = __rev_ad_w22;
    double const w53 = ((w51)*(w52));
    double const __rev_ad_w21 = ((w53)+(w45));
    double const w54 = __rev_ad_w21;
    double const __rev_ad_w20 = ((w9)*(w54));
    double const __rev_ad_w19 = __rev_ad_w20;
    double const w55 = __rev_ad_w19;
    double const __rev_ad_w18 = ((w5)*(w55));
    double const __rev_ad_w17 = __rev_ad_w18;
    double const w57 = ((2)/(4));
    double const w58 = __rev_ad_w17;
    double const __rev_ad_w16 = ((w57)*(w58));
    double const w59 = __rev_ad_w18;
    double const w60 = (-(1));
    double const w61 = __rev_ad_w27;
    double const w62 = ((w60)*(w61));
    double const __rev_ad_w14 = ((w59)+(w62));
    *d_r = __rev_ad_w14;
    double const __rev_ad_w13 = __rev_ad_w20;
    double const w63 = ((1)/(w12));
    double const w64 = __rev_ad_w13;
    double const __rev_ad_w12 = ((w63)*(w64));
    double const w65 = (-(w10));
    double const w66 = std::pow(w11, 2);
    double const w67 = ((w65)/(w66));
    double const w68 = __rev_ad_w12;
    double const w69 = ((w67)*(w68));
    double const w70 = ((w29)*(w35));
    double const __rev_ad_w11 = ((w69)+(w70));
    *d_K = __rev_ad_w11;
    double const w71 = ((w11)/(w66));
    double const w72 = ((w71)*(w68));
    double const w73 = ((w22)*(w46));
    double const __rev_ad_w10 = ((w72)+(w73));
    *d_S = __rev_ad_w10;
    double const __rev_ad_w9 = ((w20)*(w54));
    double const w74 = std::pow(w8, 2);
    double const w75 = ((-1)/(w74));
    double const w76 = __rev_ad_w9;
    double const __rev_ad_w8 = ((w75)*(w76));
    double const w77 = __rev_ad_w8;
    double const __rev_ad_w7 = ((w2)*(w77));
    double const w79 = std::pow(w5, -0.5);
    double const w80 = ((0.5)*(w79));
    double const w81 = __rev_ad_w7;
    double const w82 = ((w80)*(w81));
    double const w83 = ((w18)*(w55));
    double const w84 = ((w82)+(w83));
    double const w85 = __rev_ad_w24;
    double const w86 = ((w2)*(w85));
    double const w87 = ((w84)+(w86));
    double const w88 = ((w27)*(w37));
    double const __rev_ad_w5 = ((w87)+(w88));
    double const w89 = __rev_ad_w5;
    double const __rev_ad_w4 = ((-1)*(w89));
    *d_t = __rev_ad_w4;
    double const __rev_ad_w3 = __rev_ad_w5;
    *d_T = __rev_ad_w3;
    double const w90 = ((w7)*(w77));
    double const w91 = ((2)*(w2));
    double const w92 = __rev_ad_w16;
    double const w93 = ((w91)*(w92));
    double const w94 = ((w90)+(w93));
    double const w95 = ((w5)*(w85));
    double const __rev_ad_w2 = ((w94)+(w95));
    *d_vol = __rev_ad_w2;
    return value;
}
```

```c++
double black_foward(double t, double* d_t, double T, double* d_T, double r, double* d_r, double S, double* d_S, double K, double* d_K, double vol, double* d_vol){
    double __symbol_5      = ((T)-(t));
    double __symbol_6      = std::pow(__symbol_5, 0.5);
    double __symbol_7      = ((vol)*(__symbol_6));
    double __symbol_8      = ((1)/(__symbol_7));
    double __symbol_11     = ((S)/(K));
    double __symbol_12     = std::log(__symbol_11);
    double __symbol_15     = std::pow(vol, 2);
    double __symbol_16     = ((__symbol_15)/(2));
    double __symbol_17     = ((r)+(__symbol_16));
    double __symbol_18     = ((__symbol_17)*(__symbol_5));
    double __symbol_19     = ((__symbol_12)+(__symbol_18));
    double __symbol_20     = ((__symbol_8)*(__symbol_19));
    double __symbol_21     = std::pow(__symbol_20, 2);
    double __symbol_22     = ((0.5)*(__symbol_21));
    double __symbol_23     = (-(__symbol_22));
    double __symbol_24     = std::exp(__symbol_23);
    double __symbol_26     = ((__symbol_24)/(2.50663));
    double __symbol_28     = std::pow(__symbol_5, -0.5);
    double __symbol_30     = ((__symbol_28)*(-1));
    double __symbol_31     = ((0.5)*(__symbol_30));
    double __symbol_32     = ((vol)*(__symbol_31));
    double __symbol_33     = (-(__symbol_32));
    double __symbol_34     = std::pow(__symbol_7, 2);
    double __symbol_35     = ((__symbol_33)/(__symbol_34));
    double __symbol_36     = ((__symbol_35)*(__symbol_19));
    double __symbol_37     = ((__symbol_17)*(-1));
    double __symbol_38     = ((__symbol_8)*(__symbol_37));
    double __symbol_39     = ((__symbol_36)+(__symbol_38));
    double __symbol_40     = ((__symbol_26)*(__symbol_39));
    double __symbol_41     = ((__symbol_40)*(S));
    double __symbol_42     = ((vol)*(__symbol_5));
    double __symbol_43     = ((__symbol_20)-(__symbol_42));
    double __symbol_44     = std::pow(__symbol_43, 2);
    double __symbol_45     = ((0.5)*(__symbol_44));
    double __symbol_46     = (-(__symbol_45));
    double __symbol_47     = std::exp(__symbol_46);
    double __symbol_48     = ((__symbol_47)/(2.50663));
    double __symbol_49     = ((vol)*(-1));
    double __symbol_50     = ((__symbol_39)-(__symbol_49));
    double __symbol_51     = ((__symbol_48)*(__symbol_50));
    double __symbol_52     = (-(r));
    double __symbol_53     = ((__symbol_52)*(__symbol_5));
    double __symbol_54     = std::exp(__symbol_53);
    double __symbol_55     = ((K)*(__symbol_54));
    double __symbol_56     = ((__symbol_51)*(__symbol_55));
    double __symbol_57     = std::erfc(-(__symbol_43)/std::sqrt(2))/2;
    double __symbol_58     = ((__symbol_52)*(-1));
    double __symbol_59     = ((__symbol_54)*(__symbol_58));
    double __symbol_60     = ((K)*(__symbol_59));
    double __symbol_61     = ((__symbol_57)*(__symbol_60));
    double __symbol_62     = ((__symbol_56)+(__symbol_61));
    *d_t = ((__symbol_41)-(__symbol_62));
    double __symbol_63     = ((0.5)*(__symbol_28));
    double __symbol_64     = ((vol)*(__symbol_63));
    double __symbol_65     = (-(__symbol_64));
    double __symbol_66     = ((__symbol_65)/(__symbol_34));
    double __symbol_67     = ((__symbol_66)*(__symbol_19));
    double __symbol_68     = ((__symbol_8)*(__symbol_17));
    double __symbol_69     = ((__symbol_67)+(__symbol_68));
    double __symbol_70     = ((__symbol_26)*(__symbol_69));
    double __symbol_71     = ((__symbol_70)*(S));
    double __symbol_72     = ((__symbol_69)-(vol));
    double __symbol_73     = ((__symbol_48)*(__symbol_72));
    double __symbol_74     = ((__symbol_73)*(__symbol_55));
    double __symbol_75     = ((__symbol_54)*(__symbol_52));
    double __symbol_76     = ((K)*(__symbol_75));
    double __symbol_77     = ((__symbol_57)*(__symbol_76));
    double __symbol_78     = ((__symbol_74)+(__symbol_77));
    *d_T = ((__symbol_71)-(__symbol_78));
    double __symbol_79     = ((__symbol_8)*(__symbol_5));
    double __symbol_80     = ((__symbol_26)*(__symbol_79));
    double __symbol_81     = ((__symbol_80)*(S));
    double __symbol_82     = ((__symbol_48)*(__symbol_79));
    double __symbol_83     = ((__symbol_82)*(__symbol_55));
    double __symbol_84     = (-(1));
    double __symbol_85     = ((__symbol_84)*(__symbol_5));
    double __symbol_86     = ((__symbol_54)*(__symbol_85));
    double __symbol_87     = ((K)*(__symbol_86));
    double __symbol_88     = ((__symbol_57)*(__symbol_87));
    double __symbol_89     = ((__symbol_83)+(__symbol_88));
    *d_r = ((__symbol_81)-(__symbol_89));
    double __symbol_90     = std::pow(K, 2);
    double __symbol_91     = ((K)/(__symbol_90));
    double __symbol_92     = ((__symbol_91)/(__symbol_11));
    double __symbol_93     = ((__symbol_8)*(__symbol_92));
    double __symbol_94     = ((__symbol_26)*(__symbol_93));
    double __symbol_95     = ((__symbol_94)*(S));
    double __symbol_96     = std::erfc(-(__symbol_20)/std::sqrt(2))/2;
    double __symbol_97     = ((__symbol_95)+(__symbol_96));
    double __symbol_98     = ((__symbol_48)*(__symbol_93));
    double __symbol_99     = ((__symbol_98)*(__symbol_55));
    *d_S = ((__symbol_97)-(__symbol_99));
    double __symbol_100    = (-(S));
    double __symbol_101    = ((__symbol_100)/(__symbol_90));
    double __symbol_102    = ((__symbol_101)/(__symbol_11));
    double __symbol_103    = ((__symbol_8)*(__symbol_102));
    double __symbol_104    = ((__symbol_26)*(__symbol_103));
    double __symbol_105    = ((__symbol_104)*(S));
    double __symbol_106    = ((__symbol_48)*(__symbol_103));
    double __symbol_107    = ((__symbol_106)*(__symbol_55));
    double __symbol_108    = ((__symbol_57)*(__symbol_54));
    double __symbol_109    = ((__symbol_107)+(__symbol_108));
    *d_K = ((__symbol_105)-(__symbol_109));
    double __symbol_110    = (-(__symbol_6));
    double __symbol_111    = ((__symbol_110)/(__symbol_34));
    double __symbol_112    = ((__symbol_111)*(__symbol_19));
    double __symbol_113    = ((2)*(vol));
    double __symbol_114    = ((__symbol_113)*(2));
    double __symbol_116    = ((__symbol_114)/(4));
    double __symbol_117    = ((__symbol_116)*(__symbol_5));
    double __symbol_118    = ((__symbol_8)*(__symbol_117));
    double __symbol_119    = ((__symbol_112)+(__symbol_118));
    double __symbol_120    = ((__symbol_26)*(__symbol_119));
    double __symbol_121    = ((__symbol_120)*(S));
    double __symbol_122    = ((__symbol_119)-(__symbol_5));
    double __symbol_123    = ((__symbol_48)*(__symbol_122));
    double __symbol_124    = ((__symbol_123)*(__symbol_55));
    *d_vol = ((__symbol_121)-(__symbol_124));
    double __symbol_125    = ((__symbol_96)*(S));
    double __symbol_126    = ((__symbol_57)*(__symbol_55));
    return ((__symbol_125)-(__symbol_126));
}
```


How fast is it? I've got a file generated/black_0.cxx, with some profiles. initial results showned that the compiler can not simply be given neaive expressions (duplicate subexpression), and in-memory optimization is needed to beat finite difference


N|fd|ad|ad hand opt|ad three address
---|------|------|------|------
100|0.0001|0.0001|0.0000|0.0000
200|0.0003|0.0001|0.0000|0.0000
400|0.0005|0.0002|0.0001|0.0001
800|0.0010|0.0004|0.0001|0.0002
1600|0.0020|0.0009|0.0003|0.0003
3200|0.0018|0.0018|0.0005|0.0004
6400|0.0037|0.0016|0.0005|0.0006
12800|0.0073|0.0032|0.0010|0.0011
25600|0.0146|0.0064|0.0020|0.0022
51200|0.0293|0.0129|0.0039|0.0045
102400|0.0586|0.0257|0.0079|0.0090
204800|0.1171|0.0515|0.0157|0.0179
409600|0.2344|0.1030|0.0314|0.0358
819200|0.4688|0.2060|0.0628|0.0716
1638400|0.9374|0.4120|0.1257|0.1432
3276800|1.8742|0.8240|0.2521|0.2865
6553600|3.7657|1.6490|0.5028|0.5731
13107200|7.5069|3.2970|1.0054|1.1459
26214400|15.1139|6.6125|2.0190|2.3016
52428800|30.0545|13.4278|4.0232|4.6103
