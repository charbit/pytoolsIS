# this program computes an estimate of the matrices Upsilon and Xi f
# and compare to the asymptotic values
# If Q denotes the size of the slowness dictionary:
#  - Upsilon is a (2Qx2Q) matrix
#  - Xi is a (QxQ) matrix
#
from numpy import  array, ones, dot, transpose, cov, cos, sin, diag
from numpy import size, eye, pi, max, min, zeros, sum, trace, int
from numpy import  random
from fstataec import UpsilonXi

station = 1;
if station == 1:
        #=== I31KZ
        xsensor_m =  1000*array([
            [-0.059972130000000,  0.194591122000000,   0.391100000000000],
            [0.229169719000000,   0.083396195000000,   0.392100000000000],
            [0.122158887000000,  -0.206822564000000,   0.391800000000000],
            [-0.123753420000000,  -0.087843992000000,   0.390200000000000],
            [-0.026664123000000,   0.015567290000000,   0.391000000000000],
            [0.919425013000000,   0.719431175000000,   0.392400000000000],
            [0.183105453000000,  -1.103053672000000,   0.383100000000000],
            [-1.243469400000000,   0.384734446000000,   0.397600000000000]])
if station == 2:
        #=== I22
        xsensor_m = 1.0e+03 * ([
            [-0.088034341864435,  -0.095905624230955,   0.272000000000000],
            [-0.217769161454130,   1.227314002838975,   0.240000000000000],
            [1.046630508831105,  -0.508438802082452,   0.283000000000000],
            [-0.740827005512541,  -0.622969576526358,   0.246000000000000]])
if station == 3:
        #=== I22
        xsensor_m = 1.0e+03 *([
            [-0.088034341864435,  -0.095905624230955,   0.272000000000000],
            [-0.217769161454130,   1.227314002838975,   0.240000000000000],
            [1.046630508831105,  -0.508438802082452,    0.283000000000000]])
            
duration_s   = 60;
Fs_Hz        = 20;
N            = int(duration_s*Fs_Hz);
M            = size(xsensor_m,0);
Lruns        = 100000;
oneM         = ones([M,1]);
PiM          = dot(oneM, transpose(oneM))/M;
PiMortho     = eye(M)-PiM;

range_azimuth_deg = [0.0, 12.0];#[0.0706, 0.1893, 0.2031];#
La       = len(range_azimuth_deg);


range_elevation_deg = [0.];
Le = len(range_elevation_deg);

range_velocity_mps    = [340.];
Lc = len(range_velocity_mps)

UN1=zeros([Lruns,La]);
RN1=zeros([Lruns,La]);
FN1=zeros([Lruns,La]);

for ir in range(Lruns):
    xn  = random.randn(N,M);
    for ia in range(La):
        az = range_azimuth_deg[ia]*pi/180
        for ie in range(Le):
            elevation = range_elevation_deg[ie]*pi/180;
            for ic in range(Lc):
                velo         = range_velocity_mps[ic];
                theta        = array([-sin(az)*cos(elevation),
                    cos(az)*cos(elevation),sin(elevation)])/velo;
                tau1_pts = Fs_Hz *sum(xsensor_m * theta,1)
                tau1_posi  = -tau1_pts+max(tau1_pts);
                Ntau1      = int(N - min(tau1_pts) + max(tau1_pts))        
                xntilde1  = zeros([Ntau1,M]);
                for im in range(M):
                    id1    = int(tau1_posi[im]);
                    id2    = id1+N-1;
                    rangeid = range(id1,id2+1)
                    xntilde1[rangeid,im] = xn[:,im];
                    
                RR = dot(transpose(xntilde1),xntilde1)/N;
                UN1[ir,ia]=sum(RR)/M;
                RN1[ir,ia]=(trace(RR)-UN1[ir,ia])/(M-1);
                FN1[ir,ia]=UN1[ir,ia]/RN1[ir,ia];
#%%
matZ = zeros([Lruns,2*La]);
for ia in range(La):
    rangeia = range(2*ia,2*ia+2)
    matZ[:,rangeia[0]] = UN1[:,ia]
    matZ[:,rangeia[1]] = RN1[:,ia];

hatU   = cov(transpose(matZ))*N;
hatXi  = cov(transpose(FN1))*N;

U,Xi  = UpsilonXi(xsensor_m, Fs_Hz, range_azimuth_deg,
          range_elevation_deg, range_velocity_mps)
          
print hatXi
print Xi


print hatU
print U