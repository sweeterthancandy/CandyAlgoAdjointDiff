#ifndef INCLUDE_CODE_WRITER_H
#define INCLUDE_CODE_WRITER_H

#include <boost/optional.hpp>
#include "Cady.h"

namespace Cady
{

	class CodeWriter
	{
    public:
        void Emit(std::ostream& ostr, std::shared_ptr<Function> const& f)const
        {

            std::vector<std::string> arg_list;
            for(auto arg_iter = f->args_begin(), arg_end = f->args_end(); arg_iter!=arg_end; ++arg_iter)
            {
                auto const& arg = **arg_iter;
                if (arg.Kind() == FAK_Double)
                {
                    arg_list.push_back("double " + arg.Name());
                }
                else if (arg.Kind() == FAK_OptDoublePtr)
                {
                    arg_list.push_back("double* " + arg.Name() + " = nullptr");
                }
                else
                {
                    throw std::runtime_error("not impleme nted");
                }
                
            }
            
            ostr << "double " << f->FunctionName() << "(";
            for (size_t idx = 0; idx != arg_list.size(); ++idx)
            {
                ostr << (idx == 0 ? "" : ", ") << arg_list[idx];
            }
            ostr << ")\n{\n";
            for (auto const& instr : *f->IB())
            {
                instr->EmitCode(ostr);
                ostr << "\n";
            }
            ostr << "}\n";
        }
#if 0
        std::string GenerateString(std::shared_ptr<InstructionBlock> const& IB)const
        {
            std::stringstream ss;
            this->Emit(ss, IB);
            return ss.str();
        }
        void EmitToFile(std::string const& file_name, std::shared_ptr<InstructionBlock> const& IB, boost::optional<std::string> const& func_name = {})const
        {
            std::ofstream out;
            out.open(file_name);
            if (!out.is_open())
            {
                std::stringstream ss;
                ss << "unable to open " << file_name;
                throw std::runtime_error(ss.str());
            }
            this->Emit(out, IB, func_name);
        }
#endif
	};
} // end namespace Cady

#endif // INCLUDE_CODE_WRITER_H