import numpy as np

def autolabel(rects, ax, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

def splash_screen():
    print('--------------------------------------')
    print('- Rad Ray Tool 3.0 - Politecnico di Torino  ')
    print('- Author: Luca Sterpone ')
    print('                                        []         ')
    print('                                       []          ')
    print('     ######              ######       []                     ')
    print('     ##  ###             ##  ###     []                   ')
    print('     ##   ##             ##   ##    []                  ')
    print('     ##  ###             ##  ###   []                    ')
    print('     ######              ######   []                      ')
    print('     ####     ###  ####  ####    []## ##  ##                            ')
    print('     ## ##   ## ## ## ## ## ##  []# ## ####                               ')
    print('     ##  ##  ##### ## ## ##  ##[]#####  ##                             ')
    print('     ##   ## ## ## ## ## ##   [] ## ##  ##                             ')
    print('     ##    #### ## ####  ##* []#### ##  ##                           ')
    print('                          **[]                    ')
    print('   ======================*****================       ')
    print('                        *******                     ')
    print('   ======================*****================                 ')
    print('                        []***  ')
    print('                       []  *  ')
    print('  FreePDK45nm Version []           ')
    print('- ')
    print('- v0.1: loading all GDS vertices')#24.08.2018
    print('- v0.2: generation of cube values')
    print('- v0.3: visualizaion of data values')
    print('- v0.4: cumulative analysis ')#21.09.2018
    print('- v0.5: transient analysis ')
    print('- v0.7: cumulative energy computation')
    print('- v1.0: maximal voltage peak computation')
    print('- v2.0: generation of the transient pulse on GDS output layer')
    print('- v2.1: generation of the amplitude pulse distribution ')
    print('- v3.0: generation of the hspice simulation commands ')
    print('- v3.1: inclusion of the FreePDK 45nm z-section ')
    print('- v3.2: sensitivity heatmap XY view ')
    print('- v3.3: sensitivity heatmap update ')#11.11.2020
    print('--------------------------------------')

def arg_error():
    print('')
    print('Usage: ')
    print('RadRay_Tool [GDS txt filename] [Metals Values] [RadRay Energy] [T/C] [#] [V] [F] [HS]')
    print(' ')
    print('[GDS txt filename]: GDS-II Textual description of the Cell Under Analysis ')
    print('[Metal Values]: Data Values of the Metals with a Uniform Time Range ')
    print('[RadRay Energy]: Energy Profile for each Cell Layer ')
    print('[T/C]: Transient or Cumulative mode')
    print('[#] : Number of iterations ')
    print('[V] : Y for printing detailed PDF reports')
    print('[F] : V for 0 degree tilting (mimic radiation facility)')
    print('     R for random degree tilting (mimic realistic radiation environment)')
    print('     S for screen mode ')
    print('     Default Resolution 10 nm ')
    print('[HS]: Y enable the generation of the HSPICE simulation file')
    print('      Layer Number - Hspice test reference - Hspice reference port')