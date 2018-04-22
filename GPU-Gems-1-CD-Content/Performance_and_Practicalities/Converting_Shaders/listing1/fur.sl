
color fnc_diffuselgt    (

        color Cin;                            /* Light Colour */
        point Lin;                            /* Light Position       */
        point Nin;                            /* Surface Normal       */
        )

{
        color Cout = Cin;
        vector LN, NN;
        float Atten;

        /* normalize the stuff */
        LN = normalize(vector(Lin));
        NN = normalize(vector(Nin));

        /* diffuse calculation */
        Atten = max(0.0,LN.NN);

        Cout *= Atten;

        return (Cout);
}


#define luminance(c) comp(c,0)*0.299 + comp(c,1)*0.587 + comp(c,2)*0.114


surface srf_fur(
        /* Hair Shading... */

        float Ka   = 0.0287;
        float Kd   = 0.77;
        float Ks   = 1.285;
        float roughness1  = 0.008;
        float SPEC1  = 0.01;
        float roughness2  = 0.016;
        float SPEC2  = 0.003;
        float   start_spec = 0.3;
        float   end_spec = 0.95;
        float spec_size_fade  = 0.1;
        float illum_width  = 180;
        float var_fade_start = 0.005;
        float var_fade_end = 0.001;

        /* Hair Color */
        color rootcolor  = color (.9714, .9714, .9714);
        color tipcolor = color (.519, .325, .125);
        color specularcolor  = (color(1) + tipcolor) / 2;
        color static_ambient  = color (0.057,0.057,0.057);


        /* Variables Passed from the rib... */
        uniform float hair_col_var  = 0.0;
        uniform float hair_length = 0.0;
        uniform normal surface_normal  = normal 1;
        uniform float hair_id   = 0.0; /* Watch Out... Across Patches */
      
        )
{
        vector T = normalize (dPdv); /* tangent along length of hair */
        vector V = -normalize(I);    /* V is the view vector */
        color Cspec = 0, Cdiff = 0;  /* collect specular & diffuse light */
        float Kspec = Ks;
        vector nL;
        varying normal nSN = normalize( surface_normal );
        vector S = nSN^T;            /* Cross product of the tangent along the hair and surface normal */
        vector N_hair = (T^S);       /* N_hair is a normal for the hair oriented "away" from the surface */
        vector norm_hair;
        float  l = clamp(nSN.T,0,1);  /* Dot of surface_normal and T, used for blending */
        float T_Dot_nL = 0;
        float T_Dot_e = 0;
        float Alpha = 0;
        float Beta = 0;
        float Kajiya = 0;
        float darkening = 1.0;
        varying color final_c;

        /* values from light */
        uniform float nonspecular = 0;
        uniform color SpecularColor = 1;

        /* When the hair is exactly perpendicular to the surface, use the surface normal,
             when the hair is exactly tangent to the surface, use the hair normal 
             Otherwise, blend between the two normals in a linear fashion 
        */
        norm_hair = (l * nSN) + ( (1-l) * N_hair);
        norm_hair = normalize(norm_hair);


        /* Make the specular only hit in certain parts of the hair--v is
             along the length of the hair 
        */
        Kspec *= min( smoothstep( start_spec, start_spec + spec_size_fade, v),  
            1 - smoothstep( end_spec, end_spec - spec_size_fade, v ) );


        /* Loop over lights, catch highlights as if this was a thin cylinder,

             Specular illumination model from:
             James T. Kajiya and Timothy L.  Kay (1989) "Rendering Fur with Three 
             Dimensional Textures", Computer Graphics 23,3, 271-280  
        */

        illuminance (P, norm_hair, radians(illum_width)) {
                nL = normalize(L);

                T_Dot_nL = T.nL;
                T_Dot_e = T.V;
                Alpha = acos(T_Dot_nL);
                Beta = acos(T_Dot_e);

                Kajiya = T_Dot_nL * T_Dot_e + sin(Alpha) * sin(Beta);

                /* calculate diffuse component */

                /* get light source parameters */

                if ( lightsource("__nonspecular",nonspecular) == 0)
                        nonspecular = 0;
                if ( lightsource("__SpecularColor",SpecularColor) == 0)
                        SpecularColor = color 1;

                Cspec += (1-nonspecular) * SpecularColor * 
                    ((SPEC1*Cl*pow(Kajiya, 1/roughness1)) + 
                    (SPEC2*Cl*pow(Kajiya, 1/roughness2)));

                Cdiff += fnc_diffuselgt(Cl, L, norm_hair);
        }

        darkening = clamp(hair_col_var, 0, 1);

        darkening = (1 - (smoothstep( var_fade_end, var_fade_start, 
                     abs(luminance(Kd*Cdiff))) * darkening));

        final_c = mix( rootcolor, tipcolor, v ) * darkening;

        Ci =  ((Ka*ambient() + Kd*Cdiff + static_ambient) * final_c
              + ((v) * Kspec * Cspec * specularcolor));

        Ci = clamp(Ci, color 0, color 1 );

        Oi = Os;
        Ci = Oi * Ci;
}

     
