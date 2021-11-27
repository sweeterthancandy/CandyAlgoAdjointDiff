#ifndef CALL_OPTION_H
#define CALL_OPTION_H

/*

def call_option_analytic(x,K,tau,r,sigma):
    
    if tau < 0.0:
        return max(0.0,x - K)
    
    from scipy.stats import norm

    tmp0 = math.log(x/K)
    tmp1 = sigma**2/2
    
    d_plus = 1.0/sigma/math.sqrt(tau)*(tmp0 + ( r + tmp1 )*tau )
    d_minus = 1.0/sigma/math.sqrt(tau)*(tmp0 + ( r - tmp1 )*tau )
    
    return x * norm.cdf( d_plus ) - K * math.exp(-r*tau)* norm.cdf( d_minus )

*/

namespace CallOption
{

    struct CallOption {

        template<class Double>
        struct Build {
            Double Evaluate(
                Double x,
                Double K,
                Double tau,
                Double r,
                Double sigma)const
            {
                using MathFunctions::Phi;
                using MathFunctions::Exp;
                using MathFunctions::Call;
                using MathFunctions::Pow;
                using MathFunctions::If;

                Double d1 = ((1.0 / (vol * Pow(tau, 0.5))) * (Log(S / K) + (r + (Pow(vol, 2.0)) / 2) * tau));
                Double d2 = d1 - vol * tau;
                Double pv = K * Exp(-r * tau);
                Double black = Phi(d1) * S - Phi(d2) * pv;
                return black;
            }
            std::vector<std::string> Arguments()const
            {
                return { "x", "K", "tau", "r", "sigma" };
            }
            std::string Name()const
            {
                return "__CallOption";
            }

            Double EvaluateVec(std::vector<Double> const& args)
            {
                enum { NumArgs = 5 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return Evaluate(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4]);
            }
            template<class F>
            Double Invoke(F&& f, std::vector<Double> const& args)
            {
                enum { NumArgs = 5 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return f(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4]);
            }
        };
    };

    struct Module
    {
        template<class V>
        static void Reflect(V&& v)
        {
            v(CallOption());
        }
    };

} // end namespace CallOption

#endif // CALL_OPTIONP_H