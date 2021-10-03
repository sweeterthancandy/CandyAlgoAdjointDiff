#ifndef INCLUDE_CADY_CODEGEN_H
#define INCLUDE_CADY_CODEGEN_H

#include "Cady.h"
#include "Transform.h"

namespace Cady{
namespace CodeGen{

#if 0
struct StringCodeGenerator{
        void Emit(std::ostream& ss, Function const& f)const{

                Transform::FoldZero folder;

                // we have a vector [ x1, x2, ... ] which are the function 
                // parameters. 

                struct VariableInfo{
                        VariableInfo(std::string const& name)
                                : name_{name}
                        {}
                        std::string const& Name()const{ return name_; }
                        boost::optional<std::shared_ptr<Operator> > GetDiffLexical(std::string const& symbol)const{
                                auto iter = diff_map_.find(symbol);
                                if( iter == diff_map_.end() )
                                        return {};
                                return iter->second;
                        }
                        void MapDiff(std::string const& symbol,
                                    std::shared_ptr<Operator> const& value){
                                diff_map_[symbol] = value;
                        }
                private:
                        std::string name_;
                        std::unordered_map<std::string, std::shared_ptr<Operator> > diff_map_;
                };


                auto to_diff = f.Arguments();


                std::vector<std::shared_ptr<VariableInfo> > deps;
                for( auto const& arg : f.Arguments() ){
                        auto ptr = std::make_shared<VariableInfo>(arg);
                        for( auto const& inner_arg : to_diff ){
                                if( arg == inner_arg ){
                                        ptr->MapDiff(inner_arg, Constant::Make(1.0));
                                } else {
                                        ptr->MapDiff(inner_arg, Constant::Make(0.0));
                                }
                        }
                        deps.push_back(ptr);
                }



                ss << "double " << f.Name() << "(";
                for(size_t idx=0;idx!=deps.size();++idx){
                        if( idx != 0 ) 
                                ss << ", ";
                        ss << "double " << deps[idx]->Name();
                        ss << ", double* " << "d_" + deps[idx]->Name();
                }

                ss << ")\n";
                ss << "{\n";

                std::string indent = "    ";

                TemporaryAllocator temp_alloc;

                for(size_t idx=0;idx!=f.Statements().size();++idx){
                        // for each statement we need to add two calculations to the
                        // infomation
                        //      statement = expr
                        //      for each X in to-diff:
                        //        d_statement_X = D[X](expr)
                        //      
                        //    

                        //

                        auto const& stmt = f.Statements()[idx];
                        auto const& expr = stmt->Expr();
                        
                        auto stmt_dep = std::make_shared<VariableInfo>(stmt->Name());
                        
                        #if 0
                        ss << indent << "/* expr\n";
                        expr->Display(ss);
                        ss << indent << "*/\n";
                        #endif
                        ss << indent << "double " << stmt_dep->Name() << " = ";
                        expr->EmitCode(ss);
                        ss << ";\n";


                        for( auto const& d_symbol : to_diff ){
                                std::vector<std::string> subs;

                                for( auto const& info : deps ){

                                        auto temp_name = temp_alloc.Allocate();

                                         ;

                                        // \partial stmt / \partial symbol d symbol
                                        auto sub_diff = BinaryOperator::Mul(
                                                expr->Diff( info->Name() ),
                                                *info->GetDiffLexical(d_symbol));

                                        sub_diff = folder.Fold(sub_diff);

                                        if( sub_diff->Kind() == OPKind_Constant  ){
                                                auto constant = reinterpret_cast<Constant*>(sub_diff.get());
                                                if( constant->Value() == 0.0 )
                                                        continue;
                                        }


                                        #if 0
                                        ss << indent << "// \\partial " << stmt->Name() << " / \\partial " << info->Name() << " d " << info->Name() << "\n";
                                        #endif
                                        #if 0
                                        ss << indent << "/* expr\n";
                                        sub_diff->Display(ss);
                                        ss << indent << "*/\n";
                                        #endif
                                        ss << indent << "double " << temp_name << " = ";
                                        sub_diff->EmitCode(ss);
                                        ss << ";\n";

                                        subs.push_back(temp_name);
                                }


                                std::string token = "__diff_" + stmt->Name() + "_" + d_symbol;
                                stmt_dep->MapDiff( d_symbol, ExogenousSymbol::Make(token));


                                ss << indent << "double " << token << " = ";
                                if( subs.size() ){
                                        for(size_t idx=0;idx!=subs.size();++idx){
                                                if( idx != 0 )
                                                        ss << " + ";
                                                ss << subs[idx];
                                        }
                                } else {
                                        ss << "0.0";
                                }
                                ss << ";\n";
                        }
                        ss << "\n\n\n";
                        deps.emplace_back(stmt_dep);

                }
                        
                for( auto const& d_symbol : to_diff ){
                        ss << indent << "*d_" + d_symbol << " = " << reinterpret_cast<ExogenousSymbol*>(deps.back()->GetDiffLexical(d_symbol).get().get())->Name() << ";\n";
                }

                ss << indent << "return " << deps.back()->Name() << ";\n";
                ss << "}\n";

        }
};
#endif

} // end namespace CodeGen
} // end namespace Cady


#endif // INCLUDE_CADY_CODEGEN_H
