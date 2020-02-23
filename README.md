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
