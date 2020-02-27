(* ::Package:: *)

BeginPackage["TFfit`"]

TFfit::usage = "Transfer function";
TFnowiggles::usage = "No-wiggle Transfer function";

Begin["`Private`"]

omhh=0;
obhh=0;
thetacmb=0;
zequality=0;
kequality=0;
zdrag=0;
Rdrag=0;
Requality=0;
soundhorizon=0;
ksilk=0;
alphac=0;
betac=0;
alphab=0;
betab=0;
betanode=0;
kpeak=0;
soundhorizonfit=0;
alphagamma=0;

TFsetparameters[omega0hh_, fbaryon_, Tcmb_]:=Module[
	{zdragb1, zdragb2, alphaca1, alphaca2, betacb1, betacb2, alphabG, y},
	omhh = omega0hh;
    obhh = omhh*fbaryon;
    If[Tcmb<=0.0,Tcmb=2.728];
    thetacmb = Tcmb/2.7;

    zequality = 2.50*^4*omhh/thetacmb^4;
    kequality = 0.0746*omhh/thetacmb^2;

    zdragb1 = 0.313*omhh^-0.419*(1+0.607*omhh^0.674);
    zdragb2 = 0.238*omhh^0.223;
    zdrag = 1291*omhh^0.251/(1+0.659*omhh^0.828)*(1+zdragb1*obhh^zdragb2);
    
    Rdrag = 31.5*obhh/thetacmb^4*(1000/(1+zdrag));
    Requality = 31.5*obhh/thetacmb^4*(1000/zequality);

    soundhorizon = 2./3./kequality*Sqrt[6./Requality]*Log[(Sqrt[1+Rdrag]+Sqrt[Rdrag+Requality])/(1+Sqrt[Requality])];

    ksilk = 1.6*obhh^0.52*omhh^0.73*(1+(10.4*omhh)^-0.95);

    alphaca1 = (46.9*omhh)^0.670*(1+(32.1*omhh)^-0.532);
    alphaca2 = (12.0*omhh)^0.424*(1+(45.0*omhh)^-0.582);
    alphac = alphaca1^(-fbaryon)*alphaca2^(-fbaryon^3);
    
    betacb1 = 0.944/(1+(458*omhh)^-0.708);
    betacb2 = (0.395*omhh)^-0.0266;
    betac = 1.0/(1+betacb1*((1-fbaryon)^ betacb2-1));

    y = zequality/(1+zdrag);
    alphabG = y*(-6.*Sqrt[1+y]+(2.+3.*y)*Log[(Sqrt[1+y]+1)/(Sqrt[1+y]-1)]);
    alphab = 2.07*kequality*soundhorizon*(1+Rdrag)^-0.75*alphabG;

    betanode = 8.41*omhh^0.435;
    betab = 0.5+fbaryon+(3.-2.*fbaryon)*Sqrt[(17.2*omhh)^2.0+1];

    kpeak = 2.5*3.14159*(1+0.217*omhh)/soundhorizon;
    soundhorizonfit = 44.5*Log[9.83/omhh]/Sqrt[1+10.0*obhh^0.75];

    alphagamma = 1-0.328*Log[431.0*omhh]*fbaryon + 0.38*Log[22.3*omhh]*fbaryon^2;
]
	
	

TFfitonek[k_]:=Module[
	{Tclnbeta, Tclnnobeta, TcCalpha, TcCnoalpha, q, xx, xxtilde, qeff, Tcf, Tc, stilde, TbT0, Tb, fbaryon, Tfull, T0L0, T0C0, T0, gammaeff, TnowigglesL0, TnowigglesC0, Tnowiggles},
	(*k = Abs[k];*)
	If[k==0,Return[{1.0, 1.0, 1.0}]];
    q = k/13.41/kequality;
    xx = k*soundhorizon;

    Tclnbeta = Log[2.718282+1.8*betac*q];
    Tclnnobeta = Log[2.718282+1.8*q];
    TcCalpha = 14.2/alphac + 386.0/(1+69.9*q^1.08);
    TcCnoalpha = 14.2 + 386.0/(1+69.9*q^1.08);

    Tcf = 1.0/(1.0+(xx/5.4)^4);
    Tc = Tcf*Tclnbeta/(Tclnbeta+TcCnoalpha*q^2) +(1-Tcf)*Tclnbeta/(Tclnbeta+TcCalpha*q^2);
    
    stilde = soundhorizon*(1+(betanode/xx)^3)^(-1./3.);
    xxtilde = k*stilde;

    TbT0 = Tclnnobeta/(Tclnnobeta+TcCnoalpha*q^2);
    Tb = Sin[xxtilde]/(xxtilde)*(TbT0/(1+(xx/5.2)^2)+alphab/(1+(betab/xx)^3)*Exp[-(k/ksilk)^1.4]);
    
    fbaryon = obhh/omhh;
    Tfull = fbaryon*Tb + (1-fbaryon)*Tc;
    Return[{Tfull,Tb,Tc}];
]



TFsoundhorizonfit[omega0_, fbaryon_, hubble_]:=Module[
	{omhh, soundhorizonfitmpc},
    omhh = omega0*hubble*hubble;
    soundhorizonfitmpc = 44.5*Log[9.83/omhh]/Sqrt[1+10.0*(omhh*fbaryon)^0.75];
    Return[soundhorizonfitmpc*hubble];
]


TFkpeak[omega0_, fbaryon_, hubble_]:=Module[
	{omhh, kpeakmpc},
    omhh = omega0*hubble*hubble;
    kpeakmpc = 2.5*3.14159*(1+0.217*omhh)/TFsoundhorizonfit[omhh,fbaryon,1.0];
    Return[kpeakmpc/hubble];
]

TFnowiggles[omega0_, fbaryon_, hubble_, Tcmb_, khmpc_]:=Module[
	{k, omhh, thetacmb, kequality, q, xx, alphagamma, gammaeff, qeff, TnowigglesL0, TnowigglesC0},
    k = khmpc*hubble;
    omhh = omega0*hubble*hubble;
    If[Tcmb<=0.0, Tcmb=2.728];
    thetacmb = Tcmb/2.7;
    kequality = 0.0746*omhh/thetacmb^2;
    q = k/13.41/kequality;
    xx = k*TFsoundhorizonfit[omhh, fbaryon, 1.0];

    alphagamma = 1-0.328*Log[431.0*omhh]*fbaryon + 0.38*Log[22.3*omhh]*fbaryon^2;
    gammaeff = omhh*(alphagamma+(1-alphagamma)/(1+(0.43*xx)^4));
    qeff = q*omhh/gammaeff;

    TnowigglesL0 = Log[2.0*2.718282+1.8*qeff];
    TnowigglesC0 = 14.2 + 731.0/(1+62.5*qeff);
    Return[TnowigglesL0/(TnowigglesL0+TnowigglesC0*qeff^2)];
]


TFzerobaryon[omega0_, hubble_, Tcmb_, khmpc_]:=Module[
	{k, omhh, thetacmb, kequality, q, T0L0, T0C0},
    k = khmpc*hubble;
    omhh = omega0*hubble*hubble;
    If[Tcmb<=0.0, Tcmb=2.728];
    thetacmb = Tcmb/2.7;

    kequality = 0.0746*omhh/thetacmb^2;
    q = k/13.41/kequality;

    T0L0 = Log[2.0*2.718282+1.8*q];
    T0C0 = 14.2 + 731.0/(1+62.5*q);
    Return[T0L0/(T0L0+T0C0*q*q)];
]


TFfit[omega0_, fbaryon_, hubble_, Tcmb_, k_]:=Module[
	{},
	TFsetparameters[omega0*hubble*hubble, fbaryon, Tcmb];
	Return[TFfitonek[k*hubble]]
]


End[]
EndPackage[]
