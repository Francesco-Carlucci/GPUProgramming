import numpy as np

MAX_CUBE = 300

#DEFINITION OF THE LAYERS TECHNOLOGICAL HEIGHT and THICKNESS - 45nm FreePDK
#[height, thickness]
layers_tech = np.zeros((300,2)) 
#layers_tech[255][0]= -900
#layers_tech[255][1]= 1200
layers_tech[1][0]= 0 #N-Well -300
layers_tech[1][1]= 520 #520
layers_tech[2][0]= 220 #P-Plus
layers_tech[2][1]= 100
layers_tech[3][0]= 220 #N-Plus
layers_tech[3][1]= 110
layers_tech[4][0]= 320 #Active
layers_tech[4][1]= 20
layers_tech[5][0]= 320 #Poly 
layers_tech[5][1]= 100
layers_tech[9][0]= 620 
layers_tech[9][1]= 150
layers_tech[10][0]= 770 #Via-1
layers_tech[10][1]= 200
layers_tech[11][0]= 970 #Metal-1
layers_tech[11][1]= 250
layers_tech[12][0]= 1220 #Via-2
layers_tech[12][1]= 200
layers_tech[13][0]= 1420 #Metal-2
layers_tech[13][1]= 250

layers_cube = np.zeros ((250,MAX_CUBE,100,6))
cube_layer_id = np.zeros ((MAX_CUBE))
layer_list = []
vertex_cube_list = []

cu_layer = 0

def read_gds(path):
    cnt = 0
    en_xy = 0
    index_cube = 0
    index_vertex = 0
    f = open(path,"r")
    for line in f:
        line = line.strip()
        if line == 'BOUNDARY':
            in_b = 1
            cnt = cnt + 1
            while (in_b == 1):
                #print ('In boundary - ', cnt)
                data = f.readline() #Readling LAYER ID
                check = data.split(' ')
                data = data.strip()
                #print(data)
                #print(check)
                #Layer Storage
                if check[0] == 'LAYER':
                    cu_layer = int(check[1])
                    #add_lay = 1
                    #for item in layer_list:
                    #	if item == cu_layer:
                    #		add_lay = 0
                    #if add_lay == 1:
                    layer_list.append(cu_layer)

                if data == 'ENDEL':
                    #A cube is closed
                    vertex_cube_list.append(index_vertex)
                    #print(layers_cube)
                    index_cube = index_cube + 1
                    index_vertex = 0
                    #print(index_vertex)
                    en_xy = 0
                    in_b = 0
                    #d = input()
                    #break

                if en_xy == 1:
                    #Inserting coordinates once already recognized XY
                    #print('-- Loading layer ', cu_layer)
                        
                    layers_cube[cu_layer][index_cube][index_vertex][0] = check[0].replace(":","")
                    layers_cube[cu_layer][index_cube][index_vertex][1] = check[1].replace("\n","")
                    layers_cube[cu_layer][index_cube][index_vertex][2] = layers_tech[cu_layer][0]
                    layers_cube[cu_layer][index_cube][index_vertex][3] = check[0].replace(":","")
                    layers_cube[cu_layer][index_cube][index_vertex][4] = check[1].replace("\n","")
                    layers_cube[cu_layer][index_cube][index_vertex][5] = layers_tech[cu_layer][0] + layers_tech[cu_layer][1]
                    index_vertex = index_vertex + 1


                if check[0] == 'XY':
                    layers_cube[cu_layer][index_cube][index_vertex][0] = check[1].replace(":","")
                    layers_cube[cu_layer][index_cube][index_vertex][1] = check[2].replace("\n","")
                    layers_cube[cu_layer][index_cube][index_vertex][2] = layers_tech[cu_layer][0]
                    layers_cube[cu_layer][index_cube][index_vertex][3] = check[1].replace(":","")
                    layers_cube[cu_layer][index_cube][index_vertex][4] = check[2].replace("\n","")
                    layers_cube[cu_layer][index_cube][index_vertex][5] = layers_tech[cu_layer][0] + layers_tech[cu_layer][1]
                    index_vertex = index_vertex + 1
                    en_xy = 1

    total_cube = index_cube		

    print('-- Total Number of Cubes : ', total_cube)
    print('-- Done')
    
    return layers_cube, layer_list, vertex_cube_list

def generate_cubes(vertex_cube_list, print_list, ax_1):
    index_cube = 0
    print(print_list)
    for item in layer_list:
        if (index_cube in print_list):
            cube_layer_id[index_cube] = item
            n = 0
            xline_1 = []
            yline_1 = []
            zline_1 = []
            xline_2 = []
            yline_2 = []
            zline_2 = []
            while n < vertex_cube_list[index_cube]:
                xline_1.append(layers_cube[item][index_cube][n][0])
                yline_1.append(layers_cube[item][index_cube][n][1])
                zline_1.append(layers_cube[item][index_cube][n][2])
                xline_2.append(layers_cube[item][index_cube][n][3])
                yline_2.append(layers_cube[item][index_cube][n][4])
                zline_2.append(layers_cube[item][index_cube][n][5])
                
                ax_1.plot3D([xline_1[n],xline_2[n]], [yline_1[n],yline_2[n]], [zline_1[n],zline_2[n]], 'gray')  #'gray'
                n = n + 1
            
            ax_1.plot3D(xline_1, yline_1, zline_1, 'blue')
            ax_1.plot3D(xline_2, yline_2, zline_2, 'blue')
            n = 0
        index_cube = index_cube + 1