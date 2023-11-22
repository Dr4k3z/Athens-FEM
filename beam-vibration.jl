deltat = 0.01
tmax = 1

# Beam parameters
L = 1
h = 0.05
E = 200e9 ; nu = 0.3;  rho = 700
l = 10; ty = .3;  tz = .1           
Iyy = ty * tz^3 / 12  
Izz = tz * ty^3 / 12

ts = 0:deltat:tmax   # times vector
xs = 0:h:L  # beam mesh
ns = 1:10   # modes

wnY = ( (ns*pi).^2 ) * sqrt(E*Izz/rho/(ty*tz)/(l^4)) # Natural frecuency direction Y
wnZ = ( (ns*pi).^2 ) * sqrt(E*Iyy/rho/(ty*tz)/(l^4)) # Natural frecuency direction Z

analyticDisY = 0; analyticDisZ = 0;
analySolPos = appNodePos ;  
for i=1:length(ns)
  analyticDisY = analyticDisY + (1/(wnY(i)^2 - w^2)) * sin(i*analySolPos/l*pi) * sin(i*pi*appNodePos/l) * sin(w*ts)
  analyticDisZ = analyticDisZ + (1/(wnZ(i)^2 - w^2)) * sin(i*analySolPos/l*pi) * sin(i*pi*appNodePos/l) * sin(w*ts)
end
analyticDisY = analyticDisY * (2*Fo/(rho*ty*tz*l));
analyticDisZ = analyticDisZ * (2*Fo/(rho*ty*tz*l));