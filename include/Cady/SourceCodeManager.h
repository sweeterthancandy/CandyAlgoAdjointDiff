#ifndef CADY_SOURCE_CODE_MANAGER_H
#define CADY_SOURCE_CODE_MANAGER_H

#include <string>

namespace Cady
{


    /*
    Need a way to generate different families of functions
        1) Verbatim
        2) Three Address
        3) Forward Diff
        4) backward Diff
        */

    enum DiffMode
    {
        DM_None,
        DM_Forward,
        DM_Backward,
    };

    enum CallingConvention
    {
        CC_ReturnScalar,
        CC_ReturnArrayOfDiffs,
    };

    struct SourceCodeManager
    {
        virtual ~SourceCodeManager() = default;
        virtual std::string ImpliedName(std::string const& base_name)const = 0;
        virtual CallingConvention GetCC()const = 0;
        virtual DiffMode GetDiffMode()const = 0;
    };




    struct VerbatumSCM : SourceCodeManager
    {
        virtual std::string ImpliedName(std::string const& base_name)const
        {
            return base_name + "_verbatum";
        }
        virtual CallingConvention GetCC()const
        {
            return CC_ReturnScalar;
        }
        virtual DiffMode GetDiffMode()const
        {
            return DM_None;
        }
    };

    struct ForwardDiffSCM : SourceCodeManager
    {
        virtual std::string ImpliedName(std::string const& base_name)const
        {
            return base_name + "_fwd_diff";
        }
        virtual CallingConvention GetCC()const
        {
            return CC_ReturnArrayOfDiffs;
        }
        virtual DiffMode GetDiffMode()const
        {
            return DM_Forward;
        }
    };

    struct BackwardDiffSCM : SourceCodeManager
    {
        virtual std::string ImpliedName(std::string const& base_name)const
        {
            return base_name + "_bck_diff";
        }
        virtual CallingConvention GetCC()const
        {
            return CC_ReturnArrayOfDiffs;
        }
        virtual DiffMode GetDiffMode()const
        {
            return DM_Backward;
        }
    };


} // end namesapce Cady

#endif // CADY_SOURCE_CODE_MANAGER_H