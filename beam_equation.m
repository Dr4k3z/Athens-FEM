function beam_equation
    I = linspace(0,1,10);
    solinit = bvpinit(I,[0,0,0,0]);
    sol = bvp4c(@mat4ode,@mat4bc,solinit);  
    x = linspace(0,1);
    y = deval(sol,x);
    figure;
    subplot(2,2,1);
    plot(x,y(1,:)); grid on;
    subplot(2,2,2);
    plot(x,y(2,:)); grid on;
    subplot(2,2,3);
    plot(x,y(3,:)); grid on;
    subplot(2,2,4);
    plot(x,y(4,:)); grid on;
end

function dxdy = mat4ode(~,y)
    dxdy=[y(2) y(3) y(4) 0.1];
end

function res = mat4bc(ya,yb)
    res=[ya(1) yb(1) yb(3) ya(3)];
end

