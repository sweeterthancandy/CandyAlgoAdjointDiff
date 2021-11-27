sweeterthancady => candy algorithmic differentiation
====================================================

The idea of this is to use a DSL in C++ to generate code for an AAD function.
Take a simple python function to analytically value of KO call option.
I can compose AAD function together, as well as have if statements in the code

```python
def ko_call_option_analytic(x, K, tau, r, sigma, barrier) :
    """
    From Shreve
    """
    import math
    import scipy.stats
    def N(x) :
    return scipy.stats.norm.cdf(x)

    def factor_plus(tau, s) :
    return 1 / sigma / math.sqrt(tau) * (math.log(s) + (r + sigma * *2 / 2) * tau)

    def factor_minus(tau, s) :
    return 1 / sigma / math.sqrt(tau) * (math.log(s) + (r - sigma * *2 / 2) * tau)

    tmp0 = x * (N(factor_plus(tau, x / K)) - N(factor_plus(tau, x / barrier)))
    tmp1 = -math.exp(-r * tau) * K * (N(factor_minus(tau, x / K)) - N(factor_minus(tau, x / barrier)))
    tmp2 = -barrier * (x / barrier) * *(-2 * r / sigma * *2) * (N(factor_plus(tau, barrier * *2 / K / x)) - N(factor_plus(tau, barrier / x)))
    tmp3_inner = N(factor_minus(tau, barrier * *2 / K / x)) - N(factor_minus(tau, barrier / x))
    tmp3 = math.exp(-r * tau) * K * (x / barrier) * *(-2 * r / sigma * *2 + 1) * tmp3_inner
    return tmp0 + tmp1 + tmp2 + tmp3
```

I can write the above function in C++ in a special way, which can be consumed to generate AAD code.
My analysis showed took about 2-3X to execute the AAD version


```c++
    struct FactorPlus {

        template<class Double>
        struct Build {
            Double Evaluate(
                Double sigma,
                Double r,
                Double tau,
                Double s)const
            {
                using MathFunctions::Pow;
                using MathFunctions::Log;
                return 1 / sigma / Pow(tau, 0.5) * (Log(s) + (r + Pow(sigma, 2.0) / 2) * tau);
            }
            ...
        };
    };

    struct FactorMinus{

        template<class Double>
        struct Build {
            Double Evaluate(
                Double sigma,
                Double r,
                Double tau,
                Double s)const
            {
                using MathFunctions::Pow;
                using MathFunctions::Log;
                return 1 / sigma / Pow(tau, 0.5) * (Log(s) + (r - Pow(sigma, 2.0) / 2) * tau);
            }
            ...
        };
    };
    struct KoBarrierCallOption {

        template<class Double>
        struct Build {
            Double Evaluate(
                Double x,
                Double K,
                Double tau,
                Double r,
                Double sigma,
                Double B)const
            {
                using MathFunctions::Phi;
                using MathFunctions::Exp;
                using MathFunctions::Call;
                using MathFunctions::Pow;
                
                Double tmp0 = x * (Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, x / K)) - Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, x / B)));
                Double tmp1 = -Exp(-r * tau) * K * (Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, x / K)) - Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, x / B)));
                Double tmp2 = -B * Pow(x / B, -2.0 * r / Pow(sigma,2.0)) * (Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, Pow(B, 2.0) / K / x)) - Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, B / x)));
                Double tmp3_inner = Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, Pow(B, 2.0) / K / x)) - Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, B / x));
                Double tmp3 = Exp(-r * tau) * K * Pow(x / B, -2 * r / Pow(sigma,2.0) + 1) * tmp3_inner;
                Double pv = tmp0 + tmp1 + tmp2 + tmp3;
                return pv;
            }
            ...
        };
    };
```


From this we generate the below code, which has adjoint derivitves, which computes about 2x as slow without the derivivates (which is good)

```c++
auto __FactorPlus(const double sigma, const double r, const double tau, const double s)
{
    const double __symbol_3 = sigma;
    const double __symbol_15 = 1.000000 / __symbol_3;
    const double __symbol_1 = tau;
    const double __symbol_13 = std::pow(__symbol_1, 0.500000);
    const double __symbol_16 = __symbol_15 / __symbol_13;
    const double __symbol_9 = s;
    const double __symbol_10 = std::log(__symbol_9);
    const double __symbol_6 = r;
    const double __symbol_4 = std::pow(__symbol_3, 2.000000);
    const double __symbol_5 = __symbol_4 / 2.000000;
    const double __symbol_7 = __symbol_6 + __symbol_5;
    const double __symbol_8 = __symbol_7 * __symbol_1;
    const double __symbol_11 = __symbol_10 + __symbol_8;
    const double __symbol_17 = __symbol_16 * __symbol_11;
    const double __statement_0 = __symbol_17;
    const double result = __statement_0;
    const double __adj7 = __symbol_3;
    const double __adj18 = 2.000000 * __adj7;
    const double __adj16 = 2.000000 / 4.000000;
    const double __adj13 = __symbol_1;
    const double __adj12 = __symbol_16;
    const double __adj14 = __adj13 * __adj12;
    const double __adj17 = __adj16 * __adj14;
    const double __adj19 = __adj18 * __adj17;
    const double __adj8 = std::pow(__adj7, 2.000000);
    const double __adj10 = -1.000000 / __adj8;
    const double __adj3 = __symbol_13;
    const double __adj4 = std::pow(__adj3, 2.000000);
    const double __adj5 = __adj3 / __adj4;
    const double __adj1 = __symbol_11;
    const double __adj6 = __adj5 * __adj1;
    const double __adj11 = __adj10 * __adj6;
    const double result_d_sigma = __adj19 + __adj11;
    const double result_d_r = __adj13 * __adj12;
    const double __adj29 = __symbol_7;
    const double __adj30 = __adj29 * __adj12;
    const double __adj25 = std::pow(__adj13, -0.500000);
    const double __adj27 = 0.500000 * __adj25;
    const double __adj20 = __symbol_15;
    const double __adj21 = -__adj20;
    const double __adj22 = __adj21 / __adj4;
    const double __adj23 = __adj22 * __adj1;
    const double __adj28 = __adj27 * __adj23;
    const double result_d_tau = __adj30 + __adj28;
    const double __adj31 = __symbol_9;
    const double __adj33 = 1.000000 / __adj31;
    const double result_d_s = __adj33 * __adj12;
    return std::array<double, 5>{result, result_d_sigma, result_d_r, result_d_tau, result_d_s};
}
auto __FactorMinus(const double sigma, const double r, const double tau, const double s)
{
    const double __symbol_3 = sigma;
    const double __symbol_15 = 1.000000 / __symbol_3;
    const double __symbol_1 = tau;
    const double __symbol_13 = std::pow(__symbol_1, 0.500000);
    const double __symbol_16 = __symbol_15 / __symbol_13;
    const double __symbol_9 = s;
    const double __symbol_10 = std::log(__symbol_9);
    const double __symbol_6 = r;
    const double __symbol_4 = std::pow(__symbol_3, 2.000000);
    const double __symbol_5 = __symbol_4 / 2.000000;
    const double __symbol_7 = __symbol_6 - __symbol_5;
    const double __symbol_8 = __symbol_7 * __symbol_1;
    const double __symbol_11 = __symbol_10 + __symbol_8;
    const double __symbol_17 = __symbol_16 * __symbol_11;
    const double __statement_1 = __symbol_17;
    const double result = __statement_1;
    const double __adj7 = __symbol_3;
    const double __adj19 = 2.000000 * __adj7;
    const double __adj17 = 2.000000 / 4.000000;
    const double __adj13 = __symbol_1;
    const double __adj12 = __symbol_16;
    const double __adj14 = __adj13 * __adj12;
    const double __adj15 = -1.000000 * __adj14;
    const double __adj18 = __adj17 * __adj15;
    const double __adj20 = __adj19 * __adj18;
    const double __adj8 = std::pow(__adj7, 2.000000);
    const double __adj10 = -1.000000 / __adj8;
    const double __adj3 = __symbol_13;
    const double __adj4 = std::pow(__adj3, 2.000000);
    const double __adj5 = __adj3 / __adj4;
    const double __adj1 = __symbol_11;
    const double __adj6 = __adj5 * __adj1;
    const double __adj11 = __adj10 * __adj6;
    const double result_d_sigma = __adj20 + __adj11;
    const double result_d_r = __adj13 * __adj12;
    const double __adj30 = __symbol_7;
    const double __adj31 = __adj30 * __adj12;
    const double __adj26 = std::pow(__adj13, -0.500000);
    const double __adj28 = 0.500000 * __adj26;
    const double __adj21 = __symbol_15;
    const double __adj22 = -__adj21;
    const double __adj23 = __adj22 / __adj4;
    const double __adj24 = __adj23 * __adj1;
    const double __adj29 = __adj28 * __adj24;
    const double result_d_tau = __adj31 + __adj29;
    const double __adj32 = __symbol_9;
    const double __adj34 = 1.000000 / __adj32;
    const double result_d_s = __adj34 * __adj12;
    return std::array<double, 5>{result, result_d_sigma, result_d_r, result_d_tau, result_d_s};
}
auto __KoBarrierCallOption(const double x, const double K, const double tau, const double r, const double sigma, const double B)
{
    const double __symbol_4 = x;
    const double __symbol_1 = sigma;
    const double __symbol_2 = r;
    const double __symbol_3 = tau;
    const double __symbol_9 = K;
    const double __symbol_45 = __symbol_4 / __symbol_9;
    auto __call_result_13427076 = __FactorPlus(__symbol_1, __symbol_2, __symbol_3, __symbol_45);
    const double __symbol_55 = __call_result_13427076[0];
    const double d_13351364___symbol_1 = __call_result_13427076[1];
    const double d_13351364___symbol_2 = __call_result_13427076[2];
    const double d_13351364___symbol_3 = __call_result_13427076[3];
    const double d_13351364___symbol_45 = __call_result_13427076[4];
    const double __symbol_56 = std::erfc(-(__symbol_55) / std::sqrt(2)) / 2;
    const double __symbol_5 = B;
    const double __symbol_24 = __symbol_4 / __symbol_5;
    auto __call_result_13423716 = __FactorPlus(__symbol_1, __symbol_2, __symbol_3, __symbol_24);
    const double __symbol_53 = __call_result_13423716[0];
    const double d_13351484___symbol_1 = __call_result_13423716[1];
    const double d_13351484___symbol_2 = __call_result_13423716[2];
    const double d_13351484___symbol_3 = __call_result_13423716[3];
    const double d_13351484___symbol_24 = __call_result_13423716[4];
    const double __symbol_54 = std::erfc(-(__symbol_53) / std::sqrt(2)) / 2;
    const double __symbol_57 = __symbol_56 - __symbol_54;
    const double __symbol_58 = __symbol_4 * __symbol_57;
    const double __statement_2 = __symbol_58;
    const double __symbol_26 = -__symbol_2;
    const double __symbol_27 = __symbol_26 * __symbol_3;
    const double __symbol_28 = std::exp(__symbol_27);
    const double __symbol_49 = -__symbol_28;
    const double __symbol_50 = __symbol_49 * __symbol_9;
    auto __call_result_13563092 = __FactorMinus(__symbol_1, __symbol_2, __symbol_3, __symbol_45);
    const double __symbol_46 = __call_result_13563092[0];
    const double d_13351604___symbol_1 = __call_result_13563092[1];
    const double d_13351604___symbol_2 = __call_result_13563092[2];
    const double d_13351604___symbol_3 = __call_result_13563092[3];
    const double d_13351604___symbol_45 = __call_result_13563092[4];
    const double __symbol_47 = std::erfc(-(__symbol_46) / std::sqrt(2)) / 2;
    auto __call_result_13564172 = __FactorMinus(__symbol_1, __symbol_2, __symbol_3, __symbol_24);
    const double __symbol_43 = __call_result_13564172[0];
    const double d_13349684___symbol_1 = __call_result_13564172[1];
    const double d_13349684___symbol_2 = __call_result_13564172[2];
    const double d_13349684___symbol_3 = __call_result_13564172[3];
    const double d_13349684___symbol_24 = __call_result_13564172[4];
    const double __symbol_44 = std::erfc(-(__symbol_43) / std::sqrt(2)) / 2;
    const double __symbol_48 = __symbol_47 - __symbol_44;
    const double __symbol_51 = __symbol_50 * __symbol_48;
    const double __statement_3 = __symbol_51;
    const double __symbol_60 = __statement_2 + __statement_3;
    const double __symbol_39 = -__symbol_5;
    const double __symbol_21 = -2.000000 * __symbol_2;
    const double __symbol_19 = std::pow(__symbol_1, 2.000000);
    const double __symbol_22 = __symbol_21 / __symbol_19;
    const double __symbol_38 = std::pow(__symbol_24, __symbol_22);
    const double __symbol_40 = __symbol_39 * __symbol_38;
    const double __symbol_11 = std::pow(__symbol_5, 2.000000);
    const double __symbol_12 = __symbol_11 / __symbol_9;
    const double __symbol_13 = __symbol_12 / __symbol_4;
    auto __call_result_13315500 = __FactorPlus(__symbol_1, __symbol_2, __symbol_3, __symbol_13);
    const double __symbol_35 = __call_result_13315500[0];
    const double d_13352444___symbol_1 = __call_result_13315500[1];
    const double d_13352444___symbol_2 = __call_result_13315500[2];
    const double d_13352444___symbol_3 = __call_result_13315500[3];
    const double d_13352444___symbol_13 = __call_result_13315500[4];
    const double __symbol_36 = std::erfc(-(__symbol_35) / std::sqrt(2)) / 2;
    const double __symbol_6 = __symbol_5 / __symbol_4;
    auto __call_result_13698804 = __FactorPlus(__symbol_1, __symbol_2, __symbol_3, __symbol_6);
    const double __symbol_33 = __call_result_13698804[0];
    const double d_13351724___symbol_1 = __call_result_13698804[1];
    const double d_13351724___symbol_2 = __call_result_13698804[2];
    const double d_13351724___symbol_3 = __call_result_13698804[3];
    const double d_13351724___symbol_6 = __call_result_13698804[4];
    const double __symbol_34 = std::erfc(-(__symbol_33) / std::sqrt(2)) / 2;
    const double __symbol_37 = __symbol_36 - __symbol_34;
    const double __symbol_41 = __symbol_40 * __symbol_37;
    const double __statement_4 = __symbol_41;
    const double __symbol_61 = __symbol_60 + __statement_4;
    const double __symbol_29 = __symbol_28 * __symbol_9;
    const double __symbol_23 = __symbol_22 + 1.000000;
    const double __symbol_25 = std::pow(__symbol_24, __symbol_23);
    const double __symbol_30 = __symbol_29 * __symbol_25;
    auto __call_result_13699044 = __FactorMinus(__symbol_1, __symbol_2, __symbol_3, __symbol_13);
    const double __symbol_14 = __call_result_13699044[0];
    const double d_13351844___symbol_1 = __call_result_13699044[1];
    const double d_13351844___symbol_2 = __call_result_13699044[2];
    const double d_13351844___symbol_3 = __call_result_13699044[3];
    const double d_13351844___symbol_13 = __call_result_13699044[4];
    const double __symbol_15 = std::erfc(-(__symbol_14) / std::sqrt(2)) / 2;
    auto __call_result_13929580 = __FactorMinus(__symbol_1, __symbol_2, __symbol_3, __symbol_6);
    const double __symbol_7 = __call_result_13929580[0];
    const double d_13348844___symbol_1 = __call_result_13929580[1];
    const double d_13348844___symbol_2 = __call_result_13929580[2];
    const double d_13348844___symbol_3 = __call_result_13929580[3];
    const double d_13348844___symbol_6 = __call_result_13929580[4];
    const double __symbol_8 = std::erfc(-(__symbol_7) / std::sqrt(2)) / 2;
    const double __symbol_16 = __symbol_15 - __symbol_8;
    const double __statement_5 = __symbol_16;
    const double __symbol_31 = __symbol_30 * __statement_5;
    const double __statement_6 = __symbol_31;
    const double __symbol_62 = __symbol_61 + __statement_6;
    const double __statement_7 = __symbol_62;
    const double result = __statement_7;
    const double __adj70 = __symbol_5;
    const double __adj122 = -__adj70;
    const double __adj1 = __symbol_4;
    const double __adj96 = std::pow(__adj1, 2.000000);
    const double __adj123 = __adj122 / __adj96;
    const double __adj119 = d_13348844___symbol_6;
    const double __adj112 = __symbol_7;
    const double __adj113 = std::pow(__adj112, 2.000000);
    const double __adj114 = 0.500000 * __adj113;
    const double __adj115 = -__adj114;
    const double __adj116 = std::exp(__adj115);
    const double __adj117 = __adj116 / 2.506628;
    const double __adj85 = __symbol_30;
    const double __adj111 = -1.000000 * __adj85;
    const double __adj118 = __adj117 * __adj111;
    const double __adj120 = __adj119 * __adj118;
    const double __adj109 = d_13351724___symbol_6;
    const double __adj102 = __symbol_33;
    const double __adj103 = std::pow(__adj102, 2.000000);
    const double __adj104 = 0.500000 * __adj103;
    const double __adj105 = -__adj104;
    const double __adj106 = std::exp(__adj105);
    const double __adj107 = __adj106 / 2.506628;
    const double __adj75 = __symbol_40;
    const double __adj101 = -1.000000 * __adj75;
    const double __adj108 = __adj107 * __adj101;
    const double __adj110 = __adj109 * __adj108;
    const double __adj121 = __adj120 + __adj110;
    const double __adj124 = __adj123 * __adj121;
    const double __adj97 = __symbol_12;
    const double __adj98 = -__adj97;
    const double __adj99 = __adj98 / __adj96;
    const double __adj93 = d_13351844___symbol_13;
    const double __adj86 = __symbol_14;
    const double __adj87 = std::pow(__adj86, 2.000000);
    const double __adj88 = 0.500000 * __adj87;
    const double __adj89 = -__adj88;
    const double __adj90 = std::exp(__adj89);
    const double __adj91 = __adj90 / 2.506628;
    const double __adj92 = __adj91 * __adj85;
    const double __adj94 = __adj93 * __adj92;
    const double __adj83 = d_13352444___symbol_13;
    const double __adj76 = __symbol_35;
    const double __adj77 = std::pow(__adj76, 2.000000);
    const double __adj78 = 0.500000 * __adj77;
    const double __adj79 = -__adj78;
    const double __adj80 = std::exp(__adj79);
    const double __adj81 = __adj80 / 2.506628;
    const double __adj82 = __adj81 * __adj75;
    const double __adj84 = __adj83 * __adj82;
    const double __adj95 = __adj94 + __adj84;
    const double __adj100 = __adj99 * __adj95;
    const double __adj125 = __adj124 + __adj100;
    const double __adj74 = __symbol_57;
    const double __adj126 = __adj125 + __adj74;
    const double __adj71 = std::pow(__adj70, 2.000000);
    const double __adj72 = __adj70 / __adj71;
    const double __adj53 = __symbol_24;
    const double __adj62 = __symbol_23;
    const double __adj64 = std::pow(__adj53, __adj62);
    const double __adj63 = __adj62 / __adj53;
    const double __adj65 = __adj64 * __adj63;
    const double __adj60 = __symbol_29;
    const double __adj59 = __statement_5;
    const double __adj61 = __adj60 * __adj59;
    const double __adj66 = __adj65 * __adj61;
    const double __adj54 = __symbol_22;
    const double __adj56 = std::pow(__adj53, __adj54);
    const double __adj55 = __adj54 / __adj53;
    const double __adj57 = __adj56 * __adj55;
    const double __adj51 = __symbol_39;
    const double __adj50 = __symbol_37;
    const double __adj52 = __adj51 * __adj50;
    const double __adj58 = __adj57 * __adj52;
    const double __adj67 = __adj66 + __adj58;
    const double __adj48 = d_13349684___symbol_24;
    const double __adj41 = __symbol_43;
    const double __adj42 = std::pow(__adj41, 2.000000);
    const double __adj43 = 0.500000 * __adj42;
    const double __adj44 = -__adj43;
    const double __adj45 = std::exp(__adj44);
    const double __adj46 = __adj45 / 2.506628;
    const double __adj14 = __symbol_50;
    const double __adj40 = -1.000000 * __adj14;
    const double __adj47 = __adj46 * __adj40;
    const double __adj49 = __adj48 * __adj47;
    const double __adj68 = __adj67 + __adj49;
    const double __adj38 = d_13351484___symbol_24;
    const double __adj31 = __symbol_53;
    const double __adj32 = std::pow(__adj31, 2.000000);
    const double __adj33 = 0.500000 * __adj32;
    const double __adj34 = -__adj33;
    const double __adj35 = std::exp(__adj34);
    const double __adj36 = __adj35 / 2.506628;
    const double __adj30 = -1.000000 * __adj1;
    const double __adj37 = __adj36 * __adj30;
    const double __adj39 = __adj38 * __adj37;
    const double __adj69 = __adj68 + __adj39;
    const double __adj73 = __adj72 * __adj69;
    const double __adj127 = __adj126 + __adj73;
    const double __adj25 = __symbol_9;
    const double __adj26 = std::pow(__adj25, 2.000000);
    const double __adj27 = __adj25 / __adj26;
    const double __adj22 = d_13351604___symbol_45;
    const double __adj15 = __symbol_46;
    const double __adj16 = std::pow(__adj15, 2.000000);
    const double __adj17 = 0.500000 * __adj16;
    const double __adj18 = -__adj17;
    const double __adj19 = std::exp(__adj18);
    const double __adj20 = __adj19 / 2.506628;
    const double __adj21 = __adj20 * __adj14;
    const double __adj23 = __adj22 * __adj21;
    const double __adj12 = d_13351364___symbol_45;
    const double __adj4 = __symbol_55;
    const double __adj5 = std::pow(__adj4, 2.000000);
    const double __adj7 = 0.500000 * __adj5;
    const double __adj8 = -__adj7;
    const double __adj9 = std::exp(__adj8);
    const double __adj10 = __adj9 / 2.506628;
    const double __adj11 = __adj10 * __adj1;
    const double __adj13 = __adj12 * __adj11;
    const double __adj24 = __adj23 + __adj13;
    const double __adj28 = __adj27 * __adj24;
    const double result_d_x = __adj127 + __adj28;
    const double __adj142 = __symbol_28;
    const double __adj140 = __symbol_25;
    const double __adj141 = __adj140 * __adj59;
    const double __adj143 = __adj142 * __adj141;
    const double __adj136 = __symbol_11;
    const double __adj137 = -__adj136;
    const double __adj138 = __adj137 / __adj26;
    const double __adj134 = __adj1 / __adj96;
    const double __adj135 = __adj134 * __adj95;
    const double __adj139 = __adj138 * __adj135;
    const double __adj144 = __adj143 + __adj139;
    const double __adj132 = __symbol_49;
    const double __adj131 = __symbol_48;
    const double __adj133 = __adj132 * __adj131;
    const double __adj145 = __adj144 + __adj133;
    const double __adj128 = -__adj1;
    const double __adj129 = __adj128 / __adj26;
    const double __adj130 = __adj129 * __adj24;
    const double result_d_K = __adj145 + __adj130;
    const double __adj171 = d_13348844___symbol_3;
    const double __adj172 = __adj171 * __adj118;
    const double __adj169 = d_13351844___symbol_3;
    const double __adj170 = __adj169 * __adj92;
    const double __adj173 = __adj172 + __adj170;
    const double __adj167 = d_13351724___symbol_3;
    const double __adj168 = __adj167 * __adj108;
    const double __adj174 = __adj173 + __adj168;
    const double __adj165 = d_13352444___symbol_3;
    const double __adj166 = __adj165 * __adj82;
    const double __adj175 = __adj174 + __adj166;
    const double __adj163 = d_13349684___symbol_3;
    const double __adj164 = __adj163 * __adj47;
    const double __adj176 = __adj175 + __adj164;
    const double __adj161 = d_13351604___symbol_3;
    const double __adj162 = __adj161 * __adj21;
    const double __adj177 = __adj176 + __adj162;
    const double __adj159 = __symbol_26;
    const double __adj156 = __symbol_27;
    const double __adj157 = std::exp(__adj156);
    const double __adj154 = __adj25 * __adj141;
    const double __adj152 = -1.000000;
    const double __adj150 = __adj25 * __adj131;
    const double __adj153 = __adj152 * __adj150;
    const double __adj155 = __adj154 + __adj153;
    const double __adj158 = __adj157 * __adj155;
    const double __adj160 = __adj159 * __adj158;
    const double __adj178 = __adj177 + __adj160;
    const double __adj148 = d_13351484___symbol_3;
    const double __adj149 = __adj148 * __adj37;
    const double __adj179 = __adj178 + __adj149;
    const double __adj146 = d_13351364___symbol_3;
    const double __adj147 = __adj146 * __adj11;
    const double result_d_tau = __adj179 + __adj147;
    const double __adj209 = d_13348844___symbol_2;
    const double __adj210 = __adj209 * __adj118;
    const double __adj207 = d_13351844___symbol_2;
    const double __adj208 = __adj207 * __adj92;
    const double __adj211 = __adj210 + __adj208;
    const double __adj205 = d_13351724___symbol_2;
    const double __adj206 = __adj205 * __adj108;
    const double __adj212 = __adj211 + __adj206;
    const double __adj203 = d_13352444___symbol_2;
    const double __adj204 = __adj203 * __adj82;
    const double __adj213 = __adj212 + __adj204;
    const double __adj197 = __symbol_19;
    const double __adj198 = std::pow(__adj197, 2.000000);
    const double __adj199 = __adj197 / __adj198;
    const double __adj191 = std::log(__adj53);
    const double __adj194 = __adj64 * __adj191;
    const double __adj195 = __adj194 * __adj61;
    const double __adj192 = __adj56 * __adj191;
    const double __adj193 = __adj192 * __adj52;
    const double __adj196 = __adj195 + __adj193;
    const double __adj200 = __adj199 * __adj196;
    const double __adj202 = -2.000000 * __adj200;
    const double __adj214 = __adj213 + __adj202;
    const double __adj189 = d_13349684___symbol_2;
    const double __adj190 = __adj189 * __adj47;
    const double __adj215 = __adj214 + __adj190;
    const double __adj187 = d_13351604___symbol_2;
    const double __adj188 = __adj187 * __adj21;
    const double __adj216 = __adj215 + __adj188;
    const double __adj184 = __symbol_3;
    const double __adj185 = __adj184 * __adj158;
    const double __adj186 = __adj152 * __adj185;
    const double __adj217 = __adj216 + __adj186;
    const double __adj182 = d_13351484___symbol_2;
    const double __adj183 = __adj182 * __adj37;
    const double __adj218 = __adj217 + __adj183;
    const double __adj180 = d_13351364___symbol_2;
    const double __adj181 = __adj180 * __adj11;
    const double result_d_r = __adj218 + __adj181;
    const double __adj240 = d_13348844___symbol_1;
    const double __adj241 = __adj240 * __adj118;
    const double __adj238 = d_13351844___symbol_1;
    const double __adj239 = __adj238 * __adj92;
    const double __adj242 = __adj241 + __adj239;
    const double __adj236 = d_13351724___symbol_1;
    const double __adj237 = __adj236 * __adj108;
    const double __adj243 = __adj242 + __adj237;
    const double __adj234 = d_13352444___symbol_1;
    const double __adj235 = __adj234 * __adj82;
    const double __adj244 = __adj243 + __adj235;
    const double __adj231 = __symbol_1;
    const double __adj232 = 2.000000 * __adj231;
    const double __adj227 = __symbol_21;
    const double __adj228 = -__adj227;
    const double __adj229 = __adj228 / __adj198;
    const double __adj230 = __adj229 * __adj196;
    const double __adj233 = __adj232 * __adj230;
    const double __adj245 = __adj244 + __adj233;
    const double __adj225 = d_13349684___symbol_1;
    const double __adj226 = __adj225 * __adj47;
    const double __adj246 = __adj245 + __adj226;
    const double __adj223 = d_13351604___symbol_1;
    const double __adj224 = __adj223 * __adj21;
    const double __adj247 = __adj246 + __adj224;
    const double __adj221 = d_13351484___symbol_1;
    const double __adj222 = __adj221 * __adj37;
    const double __adj248 = __adj247 + __adj222;
    const double __adj219 = d_13351364___symbol_1;
    const double __adj220 = __adj219 * __adj11;
    const double result_d_sigma = __adj248 + __adj220;
    const double __adj257 = __adj134 * __adj121;
    const double __adj255 = 2.000000 * __adj70;
    const double __adj254 = __adj27 * __adj135;
    const double __adj256 = __adj255 * __adj254;
    const double __adj258 = __adj257 + __adj256;
    const double __adj251 = __symbol_38;
    const double __adj252 = __adj251 * __adj50;
    const double __adj253 = __adj152 * __adj252;
    const double __adj259 = __adj258 + __adj253;
    const double __adj249 = __adj128 / __adj71;
    const double __adj250 = __adj249 * __adj69;
    const double result_d_B = __adj259 + __adj250;
    return std::array<double, 7>{result, result_d_x, result_d_K, result_d_tau, result_d_r, result_d_sigma, result_d_B};
}
```
