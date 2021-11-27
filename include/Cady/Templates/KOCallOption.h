#ifndef KO_CALL_OPTION_H
#define KO_CALL_OPTION_H


/*
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

*/



namespace KoBarrierOption
{
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
            std::vector<std::string> Arguments()const
            {
                return { "sigma", "r", "tau", "s" };
            }
            std::string Name()const
            {
                return "__FactorPlus";
            }

            Double EvaluateVec(std::vector<Double> const& args)
            {
                enum { NumArgs = 4 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return Evaluate(
                    args[0],
                    args[1],
                    args[2],
                    args[3]);
            }
            template<class F>
            Double Invoke(F&& f, std::vector<Double> const& args)
            {
                enum { NumArgs = 4 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return f(
                    args[0],
                    args[1],
                    args[2],
                    args[3]);
            }
        };
    };

    struct FactorMinus {

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
            std::vector<std::string> Arguments()const
            {
                return { "sigma", "r", "tau", "s" };
            }
            std::string Name()const
            {
                return "__FactorMinus";
            }

            Double EvaluateVec(std::vector<Double> const& args)
            {
                enum { NumArgs = 4 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return Evaluate(
                    args[0],
                    args[1],
                    args[2],
                    args[3]);
            }
            template<class F>
            Double Invoke(F&& f, std::vector<Double> const& args)
            {
                enum { NumArgs = 4 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return f(
                    args[0],
                    args[1],
                    args[2],
                    args[3]);
            }
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
                Double tmp2 = -B * Pow(x / B, -2.0 * r / Pow(sigma, 2.0)) * (Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, Pow(B, 2.0) / K / x)) - Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, B / x)));
                Double tmp3_inner = Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, Pow(B, 2.0) / K / x)) - Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, B / x));
                Double tmp3 = Exp(-r * tau) * K * Pow(x / B, -2 * r / Pow(sigma, 2.0) + 1) * tmp3_inner;
                Double pv = tmp0 + tmp1 + tmp2 + tmp3;
                return pv;
            }
            std::vector<std::string> Arguments()const
            {
                return { "x", "K", "tau", "r", "sigma", "B" };
            }
            std::string Name()const
            {
                return "__KoBarrierCallOption";
            }

            Double EvaluateVec(std::vector<Double> const& args)
            {
                enum { NumArgs = 6 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return Evaluate(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5]);
            }
            template<class F>
            Double Invoke(F&& f, std::vector<Double> const& args)
            {
                enum { NumArgs = 6 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return f(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5]);
            }
        };
    };

    struct Module
    {
        template<class V>
        static void Reflect(V&& v)
        {
            v(FactorPlus());
            v(FactorMinus());
            v(KoBarrierCallOption());
        }
    };


} // end namespace KoBarrierOption

#endif // KO_CALL_OPTION_H