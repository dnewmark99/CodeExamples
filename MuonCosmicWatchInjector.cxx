// standard library stuff

#include "dataclasses/I3Double.h"
#include "dataclasses/physics/I3MCTree.h"
#include "dataclasses/physics/I3Particle.h"
#include "dataclasses/physics/I3MCTreeUtils.h"

#include "g4-larsim/MuonCosmicWatchInjector.h"
#include "g4-larsim/CCMParticleInjector.h"

#include "icetray/I3Frame.h"
#include "icetray/I3Units.h"
#include "icetray/I3Module.h"
#include "icetray/I3Logging.h"
#include "icetray/IcetrayFwd.h"
#include "icetray/I3ServiceBase.h"
#include "icetray/I3SingleServiceFactory.h"

#include "phys-services/I3GSLRandomService.h"

#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <random>

MuonCosmicWatchInjector::MuonCosmicWatchInjector(const I3Context& context) :
    CCMParticleInjector(context), randomServiceName_(""){
    randomService_ = I3RandomServicePtr();
    AddParameter("RandomServiceName", "Name of the random service in the context. If empty default random service will be used.", randomServiceName_);
}

void MuonCosmicWatchInjector::Configure() {
    CCMParticleInjector::Configure();
    GetParameter("RandomServiceName", randomServiceName_);
    if(randomServiceName_.empty()) {
        randomService_ = I3RandomServicePtr(new I3GSLRandomService(0));
        log_info("+ Random service: I3GSLRandomService  (default)");
    }
    else {
        randomService_ = GetContext().Get<I3RandomServicePtr>(randomServiceName_);
        if(randomService_) log_info("+ Random service: %s  (EXTERNAL)",  randomServiceName_.c_str());
        else log_fatal("No random service \"%s\" in context!", randomServiceName_.c_str());
    }
}

double MuonCosmicWatchInjector::SmithDuller(double emu, double zenith){
    double Emu = std::pow(10, emu);
    double Au = 2e9;
    double k = 2.66;
    double r = 0.76;
    double a = 0.0025; //GeV/gcm^{2}
    double y0 = 1000.0; //g/cm^{2}
    double bmu = 0.80;
    double c = 2.99e10;  //cm/s
    double tau_muon = 2.2e-6; //s
    double rho_o = 0.001205;  //g/cm^{3}
    double mass_muon = 0.1056; //GeV/c^{2}
    double Bmu = (bmu*mass_muon*y0)/(c*tau_muon*rho_o);  //GeV
    double lamb = 120.0; //120 g/cm^{2}
    double b = 0.771;
    double mass_pio = 0.139; //GeV/c^{2}
    double tau_pio = 2.6e-8; //s
    double jpio = (y0*mass_pio)/(c*tau_pio*rho_o); //GeV

    double Epio = (Emu + a*y0*((1.0/std::cos(zenith))-0.100))/(r); // GeV
    double exp = (Bmu)/((r*Epio+100*a)*std::cos(zenith)); // 1
    double in1 = a*((y0)/(std::cos(zenith))-100)/(r*Epio);  // 1
    in1 = std::min(in1, 1.0);
    double ins = 0.100*std::cos(zenith)*(1.0-in1);
    double Pmu = std::pow(ins,exp); // 1
    double p = (Au*(std::pow(Epio,-k))*Pmu*lamb*b*jpio)/(Epio*std::cos(zenith)+b*jpio); // 1/GeV*str*cm^{2}*s
    return p;
}

double MuonCosmicWatchInjector::SampleCDF(std::vector<double> Emu, std::vector<double> cdf) {
    double r = randomService_->Uniform(0, 1.0);

    // Find index using binary search
    auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
    size_t idx = std::distance(cdf.begin(), it);

    if (idx == 0) return std::pow(10.0, Emu[0]);

    // Linear interpolation
    double low_cdf = cdf[idx - 1];
    double high_cdf = cdf[idx];
    double low_x = Emu[idx - 1];
    double high_x = Emu[idx];

    double x_interp = low_x + (r - low_cdf) / (high_cdf - low_cdf) * (high_x - low_x);
    return std::pow(10.0, x_interp);
}

double MuonCosmicWatchInjector::SampleKineticEnergy(double zenith){
    std::vector<double> muon_energy_exponent;
    std::vector<double> muon_flux_pdf;
    std::vector<double> muon_flux_cdf;

    // need to get cdf of muon energy spectrum using Smith-Duller spectrum
    size_t n_entries = 1e5;
    double starting_emu = -3.0;
    double ending_emu = 2.0;
    double total_pdf = 0.0;

    for (size_t i = 0; i <= n_entries; i++){
        double this_emu = starting_emu + ((ending_emu - starting_emu) * (static_cast<double>(i) / static_cast<double>(n_entries)));
        double this_pdf = SmithDuller(this_emu, zenith);

        // save!
        muon_energy_exponent.push_back(this_emu);
        muon_flux_pdf.push_back(this_pdf);
        total_pdf += this_pdf;
    }

    // now get cdf
    double cumulative = 0.0;
    for (size_t i = 0; i < muon_flux_pdf.size(); i++){
        cumulative += muon_flux_pdf.at(i) / total_pdf;
        muon_flux_cdf.push_back(cumulative);
    }

    // now finally we can sample energy
    return SampleCDF(muon_energy_exponent, muon_flux_cdf);
}

double MuonCosmicWatchInjector::SampleZenithCosSquared(){
    while (true) {
        double theta = randomService_->Uniform(0, M_PI / 2);  // zenith in [0, π/2]
        double y = randomService_->Uniform(0, 1);
        if (y < std::pow(std::cos(theta), 2)) {
            return theta;
        }
    }
}

bool MuonCosmicWatchInjector::RayBoxIntersection(std::vector<double> ray_origin, std::vector<double> ray_direction, std::vector<double> box_min, std::vector<double> box_max){

    std::vector<double> inv_dir(3);
    std::vector<double> t0(3), t1(3), tmin(3), tmax(3);

    for (int i = 0; i < 3; ++i) {
        inv_dir[i] = 1.0 / ray_direction[i];
        t0[i] = (box_min[i] - ray_origin[i]) * inv_dir[i];
        t1[i] = (box_max[i] - ray_origin[i]) * inv_dir[i];
        tmin[i] = std::min(t0[i], t1[i]);
        tmax[i] = std::max(t0[i], t1[i]);
    }

    float t_enter = std::max({tmin[0], tmin[1], tmin[2]});
    float t_exit  = std::min({tmax[0], tmax[1], tmax[2]});

    if (t_enter > t_exit || t_exit < 0.0) {
        return false;
    } else {
        return true;
    }

}

bool MuonCosmicWatchInjector::CheckIntersection(double muon_x, double muon_y, double muon_z, double dx, double dy, double dz, int cw){

    // grab xmin, ymin, xmax, ymax for this CW
    double xmin = cosmic_watch_bounds.at(cw).at(0);
    double ymin = cosmic_watch_bounds.at(cw).at(1);
    double zmin = cosmic_watch_zmin;
    double xmax = cosmic_watch_bounds.at(cw).at(2);
    double ymax = cosmic_watch_bounds.at(cw).at(3);
    double zmax = cosmic_watch_zmax;

    // first check if muon intersects with top CW
    bool intersect_top = RayBoxIntersection({muon_x, muon_y, muon_z}, {dx, dy, dz}, {xmin, ymin, zmin}, {xmax, ymax, zmax});

    if (intersect_top) {
        // now check if muon intersects with bottom CW
        // assuming x, y are lined up, only z is different
        zmin -= cosmic_watch_zdelta;
        zmax -= cosmic_watch_zdelta;
        bool intersect_bottom = RayBoxIntersection({muon_x, muon_y, muon_z}, {dx, dy, dz}, {xmin, ymin, zmin}, {xmax, ymax, zmax});

        if (intersect_bottom){
            return true;
        }
    }

    return false;
}

I3MCTreePtr MuonCosmicWatchInjector::GetMCTree() {
    // first let's create our MC tree
    I3MCTreePtr mcTree = boost::make_shared<I3MCTree>();

    // define muon things
    I3Particle::ParticleType type = I3Particle::MuPlus;
    double muon_x = 0.0;
    double muon_y = 0.0;
    double muon_z = 0.0;
    double translated_muon_x = 0.0;
    double translated_muon_y = 0.0;
    double translated_muon_z = 500.0;
    double muon_zenith = 0.0;
    double muon_azimuth = 0.0;
    double muon_energy = 0.0;

    // let's create and fill our I3Particle
    double muon_type_rand = randomService_->Uniform(0.0, 1.0);
    if (muon_type_rand >= P_mu_plus){
        type = I3Particle::MuMinus;
    }

    I3Particle primary(type);

    // inject muon in circle above cosmic watches
    bool triggered_CWD = false;

    while (triggered_CWD == false){
        // sample CW to inject over
        int this_cw = randomService_->Integer(6); // returns 0 - 5 represnting CW we inject over

        // sample position
        double theta_pos = randomService_->Uniform(0.0, 2.0*M_PI);
        double r = std::sqrt(randomService_->Uniform(0.0, injection_plane_radius_ * injection_plane_radius_));
        muon_x = r * std::cos(theta_pos);
        muon_y = r * std::sin(theta_pos);
        muon_z = randomService_->Uniform(injection_plane_z_ - 2.0, injection_plane_z_ + 2.0);

        // add offset to x, y positions based on the cosmic watch we injected over
        muon_x += cosmic_watch_locations.at(this_cw).at(0);
        muon_y += cosmic_watch_locations.at(this_cw).at(1);

        // sample direction
        muon_zenith = SampleZenithCosSquared();
        muon_azimuth = randomService_->Uniform(0.0, 2.0*M_PI);
        double dx = std::sin(muon_zenith) * std::cos(muon_azimuth);
        double dy = std::sin(muon_zenith) * std::sin(muon_azimuth);
        double dz = -std::cos(muon_zenith);  // minus because muon goes downward

        const double eps = 1e-12;
        if (fabs(dz) < eps) continue;

        // now check intersection with cosmic watch detectors
        triggered_CWD = CheckIntersection(muon_x, muon_y, muon_z, dx, dy, dz, this_cw);

        if (!triggered_CWD) continue; // resample if no intersection

        // translate our position along direction vector
        double t = (translated_muon_z - muon_z) / dz;
        translated_muon_x = muon_x + t * dx;
        translated_muon_y = muon_y + t * dy;
    }

    // ok triggered_CWD == true so final thing left is to get the energy and save!
    muon_energy = SampleKineticEnergy(muon_zenith);

    // add units
    translated_muon_x *= I3Units::cm;
    translated_muon_y *= I3Units::cm;
    translated_muon_z *= I3Units::cm;
    muon_energy *= I3Units::GeV;

    primary.SetTime(0.0);
    primary.SetPos(translated_muon_x, translated_muon_y, translated_muon_z);
    primary.SetEnergy(muon_energy);
    primary.SetDir(muon_zenith, muon_azimuth);

    I3MCTreeUtils::AddPrimary(*mcTree, primary);

    return mcTree;
}

I3FrameObjectPtr MuonCosmicWatchInjector::GetSimulationConfiguration() {
    I3ParticlePtr primary = boost::make_shared<I3Particle>(I3Particle::MuPlus);
    primary->SetTime(0.0);
    primary->SetEnergy(0.0);
    primary->SetPos(0.0, 0.0, 0.0);
    primary->SetDir(0.0, 0.0);
    return primary;
}

I3_MODULE(MuonCosmicWatchInjector);

