// Full dynamic UGUCA comparison driver, adapted from its TPV101 benchmark.
// Original benchmark: Copyright (C) 2021 ETH Zurich (David S. Kammer),
// authors D. S. Kammer, G. Albertini and C.-Y. Ke. LGPL-3.0-or-later.
// Added: supplied neural coefficient fields, diagnostics and exact arrival tracking.
#include <limits>
#include "thera_law.hh"
#include "material.hh"
#include "uca_simple_mesh.hh"
#include "static_communicator_mpi.hh"
#include "unimat_shear_interface.hh"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
using namespace uguca;

int main(int argc,char **argv) {
  if(argc!=9) {
    std::cerr<<"Usage: rupture_driver Nx duration dt dump_interval fields.bin output_prefix pc_iterations barrier_region_halfwidth_m\n";
    return 2;
  }
  const int rank=StaticCommunicatorMPI::getInstance()->whoAmI();
  const int nx=std::stoi(argv[1]), nz=nx/2, npc=std::stoi(argv[7]);
  const double duration=std::stod(argv[2]), dt=std::stod(argv[3]), dump_dt=std::stod(argv[4]);
  const double lx=72000., lz=36000., dx=lx/nx, cs=3464., cp=6000., rho=2670.;
  const double barrier_edge=std::stod(argv[8]);
  const int nt=std::lround(duration/dt);
  const std::string prefix=argv[6];
  SimpleMesh mesh(lx,nx,lz,nz);
  THERALaw law(mesh,npc>0);
  const double mu=rho*cs*cs, lambda=rho*cp*cp-2.*mu;
  Material mat(mu*(3.*lambda+2.*mu)/(lambda+mu),lambda/(2.*(lambda+mu)),rho);
  const char *kernel_path=std::getenv("UGUCA_KERNEL_PATH");
  mat.readPrecomputedKernels(kernel_path ? std::string(kernel_path) : global_kernel_path);
  UnimatShearInterface interface(mesh,mat,law);
  if(dt>interface.getStableTimeStep()) throw std::runtime_error("Requested dt exceeds solver stability bound");
  interface.setTimeStep(dt);
  const int nn=mesh.getNbLocalNodes();
  double **coords=mesh.getLocalCoords();
  std::vector<double> fields;
  if(nn) {
    fields.resize(nx*nz*7);
    std::ifstream in(argv[5],std::ios::binary);
    in.read(reinterpret_cast<char*>(fields.data()),fields.size()*sizeof(double));
    if(!in || in.peek()!=EOF) throw std::runtime_error("Invalid binary input size (Nx*Nz*7 float64 required)");
  }
  std::vector<double> prestress(nn),nucleation(nn),oldv(nn,1e-12),work(nn,0.);
  std::vector<int> region(nn,-1);
  NodalFieldComponent arrival(mesh), arrival05(mesh), peak(mesh);
  arrival.setAllValuesTo(-1.); arrival05.setAllValuesTo(-1.); peak.setAllValuesTo(1e-12);
  auto &load=interface.getLoad().component(0);
  interface.getLoad().component(1).setAllValuesTo(-120e6);
  interface.getTop().getVelo().component(0).setAllValuesTo(.5e-12);
  for(int i=0;i<nn;++i) {
    int ix=std::lround(coords[0][i]/dx), iz=std::lround(coords[2][i]/dx);
    const double *r=&fields[7*(ix*nz+iz)];
    law.getA()(i)=r[0]; law.getB()(i)=r[1]; law.getDc()(i)=r[2]; law.fref(i)=r[3];
    law.getTheta()(i)=r[4]; load(i)=prestress[i]=r[5]; nucleation[i]=r[6];
    const double xc=coords[0][i]-lx/2., zc=coords[2][i]-lz/2.;
    if(std::abs(xc)<=15000. && std::abs(zc)<=7500.)
      region[i]=xc < -barrier_edge ? 0 : (xc > barrier_edge ? 2 : 1);
    for(int k=0;k<7;++k) if(!std::isfinite(r[k])) throw std::runtime_error("Non-finite input");
  }
  interface.initPredictorCorrector(npc);
  law.init(); interface.init(false);
  law.advance_state=true;
  interface.initDump(prefix,".",Dumper::Format::Binary);
  interface.registerDumpField("top_velo_0");
  interface.registerDumpField("top_disp_0");
  // Arrival/peak fields evolve at EVERY integration step and are saved at the end.
  interface.dump(0,0.);
  std::ofstream stats,stations,regional;
  const double station_xy[6][2]={{-9000,0},{-4000,0},{0,0},{4000,0},{8000,0},{12000,0}};
  int station_nodes[6]={-1,-1,-1,-1,-1,-1};
  if(rank==0) {
    regional.open(prefix+"_regional.csv");
    regional<<std::setprecision(17)<<"time,left_power_W,barrier_power_W,right_power_W,receiver_max_speed_m_s,receiver_ruptured_area_m2\n";
    stats.open(prefix+"_diagnostics.csv");
    stats<<std::setprecision(17)<<"time,moment_rate_Nm_s,integrated_friction_power_W,minimum_theta_s,minimum_power_W_m2,maximum_speed_m_s,maximum_transverse_speed_m_s\n";
    stations.open(prefix+"_stations.csv");
    stations<<std::setprecision(17)<<"time,A,B,C,D,E,F\n";
    for(int j=0;j<6;++j) {
      double nearest=1e100;
      for(int i=0;i<nn;++i) {
        double r=std::hypot(coords[0][i]-lx/2.-station_xy[j][0],coords[2][i]-lz/2.-station_xy[j][1]);
        if(r<nearest) {nearest=r;station_nodes[j]=i;}
      }
    }
    std::ofstream meta(prefix+"_metadata.json");
    meta<<std::setprecision(17)<<"{\"nx\":"<<nx<<",\"nz\":"<<nz<<",\"dx_m\":"<<dx
        <<",\"dt_s\":"<<dt<<",\"duration_s\":"<<nt*dt<<",\"predictor_corrector_iterations\":"<<npc
        <<",\"arrival_thresholds_m_s\":[0.001,0.5],\"stations_actual_xy_m\":[";
    for(int j=0;j<6;++j) {if(j)meta<<",";int i=station_nodes[j];meta<<"["<<coords[0][i]-lx/2.<<","<<coords[2][i]-lz/2.<<"]";}
    meta<<"]}\n";
  }
  double next_dump=dump_dt;
  const auto start=std::chrono::steady_clock::now();
  for(int s=1;s<=nt;++s) {
    const double t=s*dt;
    const double G=t<1. ? std::exp((t-1.)*(t-1.)/(t*(t-2.))) : 1.;
    for(int i=0;i<nn;++i) load(i)=prestress[i]+25e6*nucleation[i]*G;
    interface.advanceTimeStep();
    if(rank==0) {
      auto &vel=interface.getTop().getVelo().component(0);
      auto &tau=interface.getCohesion().component(0);
      double regional_power[3]={0.,0.,0.},receiver_max=0.,receiver_area=0.;
      double sumv=0.,power=0.,mintheta=1e300,minpower=1e300,maxv=0.,maxtrans=0.;
      for(int i=0;i<nn;++i) {
        const double v=2.*vel(i), speed=std::abs(v), pw=tau(i)*v;
        if(!std::isfinite(v)) throw std::runtime_error("Non-finite velocity");
        if(arrival(i)<0. && speed>=1e-3) {
          double frac=(1e-3-std::abs(oldv[i]))/(speed-std::abs(oldv[i]));
          arrival(i)=t-dt+dt*std::clamp(frac,0.,1.);
        }
        if(arrival05(i)<0. && speed>=.5) {
          double frac=(.5-std::abs(oldv[i]))/(speed-std::abs(oldv[i]));
          arrival05(i)=t-dt+dt*std::clamp(frac,0.,1.);
        }
        oldv[i]=v; peak(i)=std::max(peak(i),speed);
        work[i]+=dt*pw;
        if(region[i]>=0) regional_power[region[i]]+=pw*dx*dx;
        if(region[i]==2) {receiver_max=std::max(receiver_max,speed); if(arrival05(i)>=0.)receiver_area+=dx*dx;}
        sumv+=v;power+=pw;minpower=std::min(minpower,pw);maxv=std::max(maxv,speed);
        mintheta=std::min(mintheta,law.getTheta()(i));
        maxtrans=std::max(maxtrans,std::abs(2.*interface.getTop().getVelo().component(2)(i)));
      }
      stats<<t<<","<<mu*dx*dx*sumv<<","<<dx*dx*power<<","<<mintheta<<","<<minpower<<","<<maxv<<","<<maxtrans<<"\n";
      regional<<t<<","<<regional_power[0]<<","<<regional_power[1]<<","<<regional_power[2]<<","<<receiver_max<<","<<receiver_area<<"\n";
      stations<<t;for(int j=0;j<6;++j)stations<<","<<2.*vel(station_nodes[j]);stations<<"\n";
      if(s%100==0 || s==nt) std::cout<<prefix<<" step "<<s<<"/"<<nt<<" peakV "<<maxv<<std::endl;
    }
    if(t+dt/2.>=next_dump || s==nt) {
      if(rank==0) interface.dump(s,t);
      next_dump+=dump_dt;
    }
  }
  if(rank==0) {
    std::ofstream final(prefix+"_final_fields.bin",std::ios::binary);
    for(int i=0;i<nn;++i) {
      double values[5]={arrival(i),arrival05(i),peak(i),law.getTheta()(i),
                       2.*interface.getTop().getDisp().component(0)(i)};
      final.write(reinterpret_cast<char*>(values),sizeof(values));
    }
    std::ofstream wf(prefix+"_friction_work.bin",std::ios::binary);
    wf.write(reinterpret_cast<char*>(work.data()),work.size()*sizeof(double));
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream timing(prefix+"_wall_seconds.txt");timing<<seconds<<"\n";
    std::cout<<"Completed "<<prefix<<" in "<<seconds<<" seconds\n";
  }
  StaticCommunicatorMPI::getInstance()->finalize();
}
