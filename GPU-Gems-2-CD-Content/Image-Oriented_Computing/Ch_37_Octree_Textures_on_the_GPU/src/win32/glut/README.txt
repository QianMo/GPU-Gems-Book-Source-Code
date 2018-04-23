La fonction
/* CENTRY */
void *APIENTRY 
glutGetHandle()
{
	return ((void *)__glutCurrentWindow->win);
}
/* ENDCENTRY */
doit etre ajoutee dans glut_get.c
Ainsi que l'export dans glut.def