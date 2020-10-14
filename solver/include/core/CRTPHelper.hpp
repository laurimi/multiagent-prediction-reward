#ifndef CRTPHELPER_HPP
#define CRTPHELPER_HPP
namespace npgi
{
template <typename Derived>
struct crtp_helper
{
    Derived& underlying() { return static_cast<Derived&>(*this); }
    Derived const& underlying() const { return static_cast<Derived const&>(*this); }
};	
}


#endif // CRTPHELPER_HPP