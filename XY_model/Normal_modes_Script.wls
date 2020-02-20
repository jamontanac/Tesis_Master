#!/usr/bin/env wolframscript
(* ::Package:: *)

nn=80001;
ll=50;
prec=120;
nData=500;
\[Gamma]=1/2//N[#,prec]&;
\[Lambda]=1/2//N[#,prec]&;
a0=\[Lambda]//N[#,prec]&;
a1=(1-\[Gamma])/2//N[#,prec]&;
a2=(1+\[Gamma])/2//N[#,prec]&;
\[Alpha][\[Theta]_]=(a0+ Cos[\[Theta]]);
\[Beta][\[Theta]_]=(-a1+a2) Sin[\[Theta]];
\[Omega][\[Theta]_]=Sqrt[\[Alpha][\[Theta]]^2+\[Beta][\[Theta]]^2];
\[Phi][\[Theta]_]=ArcTan[\[Alpha][\[Theta]],\[Beta][\[Theta]]];
t=Table[k,{k,-((nn-1)/2),(nn-1)/2}];
\[Mu]=0.0//N[#,prec]&;
\[Beta]T =0.40824980415756 //N[#,prec]&;
p[l_]:=1/(Exp[\[Beta]T (\[Omega][((2 \[Pi])/nn) l]-\[Mu])]+1);
Data = ParallelTable[mCos=Table[If[RandomReal[]>p[k],-(1/2),1/2],{k,1,(nn-1)/2}];
mCos0=If[RandomReal[]>p[0],-(1/2),1/2];
mSin=Table[If[RandomReal[]>p[k],-(1/2),1/2],{k,1,(nn-1)/2}];
mSin0=If[RandomReal[]>p[0],-(1/2),1/2];
(*mCos=Table[-(1/2),{k,1,(nn-1)/2}];
mCos0=-(1/2);
mSin=Table[-(1/2),{k,1,(nn-1)/2}];
mSin0=-(1/2);*)

MPlusBand=((If[#==0,(mCos0+mSin0)/2,(mCos[[Abs[#]]]+mSin[[Abs[#]]])/2]) E^(I Sign[(2 \[Pi])/nn #] \[Phi][Abs[(2 \[Pi])/nn #]])&/@t)//RotateLeft[#,(nn-1)/2]&//Fourier;
MMinusAntiBand=((If[#==0,(mCos0-mSin0)/2,(mCos[[Abs[#]]]-mSin[[Abs[#]]])/2]) E^(I Sign[(2 \[Pi])/nn #] (\[Phi][Abs[(2 \[Pi])/nn #]]))&/@t)//RotateLeft[#,(nn-1)/2]&//Fourier;
Join[{Join[Take[MPlusBand,ll],Take[MPlusBand,-ll+1]],Take[MMinusAntiBand,2*ll-1]}],{ii,nData}];
Res=Mean[Data];
T=ToeplitzMatrix[Take[Res[[1]],ll],Join[{Res[[1]][[1]]},Take[Res[[1]],-ll+1]//Reverse]];
H= HankelMatrix[Res[[2]][[1;;ll]],(Res[[2]]//RotateLeft[#,-ll]&)[[1;;ll]]];
M=(T+H)/(Sqrt[nn]//N[#,prec]&);
S=SingularValueDecomposition[M];
P=(S[[1]]^2+S[[3]]^2)/2 //N[#,prec]&;
Export[Directory<>"Participation.csv",P]
