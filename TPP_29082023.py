######################################################################################################################################
#------------------------------------------------------importing libraries -----------------------------------------------------------
######################################################################################################################################
import streamlit as st
import numpy as np
import numpy.linalg as LA
import math as m
import matplotlib.pyplot as plt
import pandas as pd

######################################################################################################################################
st.set_page_config(page_title="Niladri's world!",page_icon=":smirk_cat:",layout="wide")
######################################################################################################################################

hide_menu_style="""
        <style>
        #MainMenu {visbility: hidden;}
        footer{visibilty:hidden;}
        </style>
        """
st.markdown(hide_menu_style,unsafe_allow_html=True)
######################################################################################################################################
#--------Header Section---------
######################################################################################################################################
st.write("""
# Structural Geology Practical NoteBook WebApp
 """)
with st.container():
    st.write("###### Hi, I am Niladri Das ! :wave:")
    st.write("######  A Post Graduate student of Applied Geology")
    st.write("######  Share Your Thoughts!:")

######################################################################################################################################
#-------------contact-----------
######################################################################################################################################
    st.write("""
Email id: niladridaschakdaha@gmail.com
 """)
st.write("""
Linkedin :https://www.linkedin.com/in/niladri-das-49945b21b/
 """)
######################################################################################################################################   
#-------------dropdown-----------
st.write("---")

# Using object notation
dropdown = st.sidebar.selectbox(
    "Choose what you want to calculate:",
    ("1 Three Point Problem Solver", "2 Calculation of Attitude of Bed from 2 Borehole Data","3 Calculation of separation of Litho-contact due to Faulting","4 Calculation of Apparent thickness of Bed","5 Calculation of True Dip from 2 Apparent dip and dip directions","6 Orientation of intersection line of 2 planes","7 Finding orientation of plane passing through 2 lines", "8 Resultant Vector Calculator")
)

######################################################################################################################################
#-------------------------------------------------1 Three Point Problem Solver--------------------------------------------------------
######################################################################################################################################

if dropdown=="1 Three Point Problem Solver":
    st.title("Three point Problem Solver:")

    st.write("""
                ##### •Instructions: If depth is given, use '-ve'sign in elevation tabs
            """)

    st.write("""
                ###### •Sample question:
            """)
    
    st.write("""
                ###### 1. The high elevation point (700m) is point “H”, the middle elevation point (500m) is “M”, and the low elevation (200m point is “L”. The bearing direction from “H” to “M” (S67W) and from “H” to “L” (S18W) must be determined with a protractor, and the map distance from “H” to “M” (4100m) and from “H” to “L” (5160m) also must be determined with a scale.

            """)
    st.write("""
                #### Provide Inputs:
                #### 
            """)

#INSTRUCTIONS:
#three_point_problem(h1,h2,h3,Maz_deg,Laz_deg,d2,d3)
#h1 = Elevation of Highest Point
#h2 = Elevation of Medium Point
#h3 = Elevation of Lowest Point
#Maz_deg = Azimuth of Medium Point with Highest point in degrees
#Laz_deg = Azimuth of Lowest Point with Highest point in degrees
#d2 = Aerial distance of Medium Point from Highest Point
#d3 = Aerial distance of Lowest Point from Highest Point


    # Create a two-column layout
    col1, col2, col3 = st.columns(3)

# Add content to the first column
    with col1:
    
        h1=st.number_input('Enter Elevation of Highest Point:')
# Add content to the second column
    with col2:
    
        h2=st.number_input('Enter Elevation of Medium Point:')

# Add content to the third column
    with col3:
    
        h3=st.number_input('Enter Elevation of Lowest Point:')
    
# Create a two-column layout
    col4,col5 = st.columns(2)

# Add content to the 4th column
    with col4:
    
        Maz_deg=st.number_input('Enter azimuth of Medium Point with Highest point (in °)')
# Add content to the 5th column
    with col5:
    
        Laz_deg=st.number_input('Enter azimuth of Lowest Point with Highest point (in °)')


# Add content to the 4th column
    with col4:
        d2=st.number_input('Enter Aerial Distance of Medium Point from Highest Point')
# Add content to the 7th column
    with col5:
        d3=st.number_input('Enter Aerial Distance of Lowest Point from Highest Point')


    x=np.array([h1,h2,h3])

    Maz=np.deg2rad(Maz_deg)
    Laz=np.deg2rad(Laz_deg)

    a=d2*(h3-h1)*np.cos(Maz)-d3*(h2-h1)*np.cos(Laz)
    b=d3*(h2-h1)*np.sin(Laz)-d2*(h3-h1)*np.sin(Maz)
    c=d2*d3*np.sin(Maz-Laz)

    normal_vector=np.array([a,b,c])
    d=LA.norm(normal_vector)

    TrueDip=(np.arccos(abs(c)/d))*(180/np.pi)


    if c>=0:                                        #corrected!dipdirection
        dip_dir_vec=np.array([a,b])
    elif c<0:
        dip_dir_vec=np.array([-a,-b])


    nv=np.array([0,1])

    inner = np.inner(dip_dir_vec, nv)
    norms = LA.norm(dip_dir_vec) * LA.norm(nv)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg1 = np.rad2deg(rad)

    if dip_dir_vec[0]<0:                 #logic for measuring angle >180
        DipDirection=360-deg1
    elif dip_dir_vec[0]>=0:
        DipDirection=deg1
    
    if DipDirection<90:
        Strike=DipDirection+90
    elif DipDirection>=90 and DipDirection<270:
        Strike=DipDirection-90
    elif DipDirection>=270:
        Strike=DipDirection+90-360
        
    ans_array=np.array([Strike,TrueDip,DipDirection])
    RO_ans_array=np.round(ans_array,3)

    st.write('Solution:')
    st.write('• Attitude of the bed is:\n',RO_ans_array[0],'°/',RO_ans_array[1],'°→',RO_ans_array[2],'°')
    
    xH=0
    yH=0     # (x,y) coordinates for H

    xM=d2*np.sin(Maz)
    yM=d2*np.cos(Maz)      # (x,y) coordinates for M

    xL=d3*np.sin(Laz)
    yL=d3*np.cos(Laz)      # (x,y) coordinates for L

    xJ=((h2-h1)/(h3-h1))*d3*np.sin(Laz)
    yJ=((h2-h1)/(h3-h1))*d3*np.cos(Laz) 

    A=min(xH,xM,xL,xJ)
    B=max(xH,xM,xL,xJ)
    C=min(yH,yM,yL,yJ) 
    D=max(yH,yM,yL,yJ)

# Define the points to plot
    points = {'H': (xH, yH), 'M': (xM, yM), 'L': (xL, yL), 'J': (xJ, yJ)}

# Create a scatter plot of the points
    fig, ax = plt.subplots()
    for name, (x, y) in points.items():
        ax.scatter(x, y, label=name)
        ax.plot([xH,xM],[yH,yM],color='black')
        ax.plot([xH,xL],[yH,yL],color='black')
        ax.plot([xM,xL],[yM,yL],color='black')
    
# Add axis labels and legend
    ax.set_xlabel('East(unit)→')
    ax.set_ylabel('North(unit)→')
    ax.legend()
    ax.set_title('Three point Problem')
    ax.plot([xM,xJ],[yM,yJ],color='red',label="strikeline")

#DipDirection vector:
    ax.quiver(((xJ+xM)/2),((yJ+yM)/2),dip_dir_vec[0],dip_dir_vec[1],color='blue',label='dipdirection')

    ax.legend()
# Name each Points:
    ax.text(xH,yH,"H")
    ax.text(xM,yM,"M")
    ax.text(xL,yL,"L")
    ax.text(xJ,yJ,"J")

#Set the plot size
    fig.set_size_inches(6, 6*((D-C)/(B-A)))

# Display the plot in Streamlit
    st.pyplot(fig)




######################################################################################################################################
#-----------------------------------2 Calculation of Attitude of Bed from 2 Borehole Data---------------------------------------------
######################################################################################################################################



elif dropdown=="2 Calculation of Attitude of Bed from 2 Borehole Data":
    st.title("Calculation of Attitude of Bed from 2 Borehole Data:")

    st.write("""
            ###### •Sample questions:
        """)

    st.write("""
            ###### 1. To explore a  Coal Bed in an area 2 horizontal boreholes [azimuth 120°  and 180°] are given.The drill cores recovered from the 2 boreholes show coal laminations with their normal orientedat an angle of 40°  and 50°  with the core axis respectively. Find out the attitude of the coal bed.
        """)
    st.write("""
            ####
        """)


    st.write("""
            ###### 2. In an area 2 inclined bore holes with orientations 60° ➞100° and 40° ➞130°   are given to explore a coal bed . Core samples in these 2 bore holes show coal lamintaions with their normal at an angle of 70° and 60° respectively. Find the possible orientation of the coal bed in this area.
        """)
    st.write("""
            #### Provide Inputs:
            #### 
        """)


    # Create a 3-column layout
    col1, col2, col3 = st.columns(3)
# Add content to the first column
    with col1:

        p1_deg=st.number_input('Plunge of axis of borehole1 (in°):')
        p2_deg=st.number_input('Plunge of axis of borehole2 (in°):')
# Add content to the second column
    with col2:

        t1_deg=st.number_input('Trend of axis of borehole1(in°):')
        t2_deg=st.number_input('Trend of axis of borehole2(in°):')

# Add content to the third column
    with col3:

        theta1_deg=st.number_input('Angle between borehole1 axis and bed normal (in°):')
        theta2_deg=st.number_input('Angle between borehole2 axis and bed normal (in°):')



        p1=np.deg2rad(p1_deg)
        t1=np.deg2rad(t1_deg)
        theta1=np.deg2rad(theta1_deg)   # taking inputs!
        p2=np.deg2rad(p2_deg)
        t2=np.deg2rad(t2_deg)
        theta2=np.deg2rad(theta2_deg)
#---------------------------------
        a1=np.cos(p1)*np.sin(t1)
        b1=np.cos(p1)*np.cos(t1)        # converting to direction cosines
        c1=-np.sin(p1)
#---------------------------------
        a2=np.cos(p2)*np.sin(t2)
        b2=np.cos(p2)*np.cos(t2)        # converting to direction cosines
        c2=-np.sin(p2)
#---------------------------------
        l1=[a1,b1,c1]
        l2=[a2,b2,c2]                   # making list to form an array
#---------------------------------
        v1=np.array(l1)                 # forming array for vectors
        v2=np.array(l2)
#---------------------------------
        dot=np.dot(v1,v2)
        phi=np.arccos(dot)
        phi_deg=np.arccos(dot)*(180/np.pi)
        cross=np.cross(v1,v2)
        v3=cross/LA.norm(cross)
#---------------------------------
        alpha=(np.cos(theta2)-(np.cos(phi)*np.cos(theta1)))/(np.cos(theta1)-(np.cos(phi)*np.cos(theta2)))
#---------------------------------
        d1=(np.tan(theta1))**2
        d2=1+2*alpha*np.cos(phi)
        d3=(np.cos(phi)**2)/((np.cos(theta1))**2)-1
#-------------------------------------
        beta=m.sqrt((d1*d2+(alpha**2)*d3))
#-------------------------------------
        x1=1*v1+alpha*v2+beta*v3
        x2=1*v1+alpha*v2-beta*v3
#-------------------------------------
        if x1[2]>0:
            x1=-x1
        elif x1[2]<=0:
            x1=x1

        if x2[2]>0:
            x2=-x2
        elif x2[2]<=0:
            x2=x2
#-------------------------------------
        nv=np.array([0,1,0])       #north vector
#-------------------------------------
        x1d=np.array([x1[0],x1[1],0])  #string
        x2d=np.array([x2[0],x2[1],0])
#------------------------------------
        def unit_vector(vector):                                                  #
            return vector / np.linalg.norm(vector)                                #
#------------------------------------
        x1d_u = unit_vector(x1d)                                                  #  Angle bet 2 vectors
        nv_u = unit_vector(nv)                                                    #
        deg1=(np.arccos(np.clip(np.dot(x1d_u,nv_u), -1.0, 1.0)))*(180/np.pi)      #

        if x1d[0]<0:
            Trend1=360-deg1
        elif x1d[0]>=0:
            Trend1=deg1

        if Trend1<90:
            Strike1=Trend1+90
        elif Trend1>=90 and Trend1<270:
            Strike1=Trend1-90
        elif Trend1>=270:
            Strike1=Trend1+90-360

        if Trend1<180:
            DipDirection1=Trend1+180
        elif Trend1>=180:
            DipDirection1=Trend1-180
#------------------------------------
        x1_u = unit_vector(x1)
        Plunge1=np.rad2deg(np.arccos(np.clip(np.dot(x1d_u, x1_u), -1.0, 1.0)))
        Dip1=90-Plunge1
#------------------------------------
#------------------------------------
        x2d_u = unit_vector(x2d)
        nv_u = unit_vector(nv)
        deg2=(np.arccos(np.clip(np.dot(x2d_u, nv_u), -1.0, 1.0)))*(180/np.pi)

        if x2d[0]<0:
            Trend2=360-deg2
        elif x2d[0]>=0:
            Trend2=deg2

        if Trend2<90:
            Strike2=Trend2+90
        elif Trend2>=90 and Trend2<270:
            Strike2=Trend2-90
        elif Trend2>=270:
            Strike2=Trend2+90-360

        if Trend2<180:
            DipDirection2=Trend2+180
        elif Trend2>=180:
            DipDirection2=Trend2-180
#------------------------------------
        x2_u = unit_vector(x2)
        Plunge2=np.rad2deg(np.arccos(np.clip(np.dot(x2d_u, x2_u), -1.0, 1.0)))
        Dip2=90-Plunge2

        ans_array1=np.array([Strike1,Dip1,DipDirection1])
        ans_array2=np.array([Strike2,Dip2,DipDirection2])
        RO_ans_array1=np.round(ans_array1,3)
        RO_ans_array2=np.round(ans_array2,3)

    st.write('Solution:')
    st.write('• One possible orientation of the bed is:\n',RO_ans_array1[0],'°/',RO_ans_array1[1],'°→',RO_ans_array1[2],'°')
    st.write('• Another possible orientation of the bed is:\n',RO_ans_array2[0],'°/',RO_ans_array2[1],'°→',RO_ans_array2[2],'°')




######################################################################################################################################
#----------------------------------------------3 Calculation of separation of Litho-contact due to Faulting---------------------------
######################################################################################################################################




elif dropdown=="3 Calculation of separation of Litho-contact due to Faulting":
    st.title("Calculation of separation of Litho-contact due to Faulting:")

    st.write("""
                ##### •Instructions: Enter strike value that is towards left of the dip direction 
            """)
    st.write("""
                ###### •Sample question:
            """)
    st.write("""
                ###### 1. A lithocontact having attitude of 30°/40°→120° is affected due to faulting. The attitude of the fault plane is 20°/68°→110°. The netslip magnitude is 27 unit and netslip picth is 70°. Calculate the strike and dip separation of the lithocontact.
            """)
    st.write("""
                #### Provide Inputs:
                #### 
            """)

    # Create a 2-column layout
    col1, col2, col3 = st.columns(3)
    # Add content to the first column
    with col1:
        sf_deg=st.number_input('Enter strike of fault plane (in °):')
        df_deg=st.number_input('Enter dip of fault plane (in °)')
    # Add content to the second column
    with col2:
        sl_deg=st.number_input('Enter strike of litho contact (in °):')
        dl_deg=st.number_input('Enter dip of litho contact (in °):')
    # Add content to the 3rd column
    with col3:
        nsp_deg=st.number_input('Enter Netslip pitch (in °)')
        nsm=st.number_input('Enter Netslip Magnitude:')


    #-------------------
    sf=np.deg2rad(sf_deg)
    df=np.deg2rad(df_deg)
    sl=np.deg2rad(sl_deg)           #degree to radian
    dl=np.deg2rad(dl_deg)
    nsp=np.deg2rad(nsp_deg)

    #--------------------
    #net slip vector:
    #--------------------

    a1=nsm*np.sin(nsp)*np.cos(df)*np.cos(sf)+nsm*np.cos(nsp)*np.sin(sf)
    a2=-nsm*np.sin(nsp)*np.cos(df)*np.sin(sf)+nsm*np.cos(nsp)*np.cos(sf)
    a3=-nsm*np.sin(nsp)*np.sin(df)
    a=np.array([a1,a2,a3])

    #---------------------------------------------------------------
    b1=np.sin(dl)*np.sin(sl)*np.cos(df)-np.sin(df)*np.sin(sf)*np.cos(dl)
    b2=np.sin(dl)*np.cos(sl)*np.cos(df)-np.sin(df)*np.cos(sf)*np.cos(dl)
    b3=np.sin(dl)*np.cos(sl)*np.sin(df)*np.sin(sf)-np.sin(df)*np.cos(sf)*np.sin(dl)*np.sin(sl)
    b=np.array([b1,b2,b3])


    #Vector pf intersection of FP and LC
    m1=np.sin(dl)*np.sin(sl)*np.cos(df)-np.sin(df)*np.sin(sf)*np.cos(dl)
    m2=np.sin(dl)*np.cos(sl)*np.cos(df)-np.sin(df)*np.cos(sf)*np.cos(dl)
    m3=np.sin(dl)*np.cos(sl)*np.sin(dl)*np.sin(sl)-np.sin(df)*np.cos(sf)*np.sin(dl)*np.sin(sl)


    # +X axis:
    x1=np.sin(df)*np.sin(sf)
    x2=np.sin(df)*np.cos(sf)
    x3=0


    l1=[m1,m2,m3]
    l2=[x1,x2,x3]


    v1=np.array(l1)   #+x axis
    v2=np.array(l2)   # intersction line

    inner = np.inner(v1, v2)
    norms = LA.norm(v1) * LA.norm(v2)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg1 = np.rad2deg(rad)


    if m3<0:
        taninv_m=180-deg1
    elif m3>=0:
        taninv_m=deg1

    m=np.tan(taninv_m)


    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    def line_equation(m, c, x):
        return m*x + c

    st.title('Line Equation Plot (Under Construction)')


    m = m=np.tan(taninv_m)
    c = 0

    m_1=0
    c_1=0


    dx1=nsm*np.cos(nsp)
    dy1=nsm*np.sin(nsp)
    v_1=np.array([dx1,dy1])

    x = np.linspace(-dx1, +dx1, 100)
    y1 = line_equation(m, c, x)
    y2 = line_equation(m_1, c_1, x)

    fig, ax = plt.subplots()
    ax.plot(x, y1,color='red',label='FW')
    ax.plot(x, y2,label='FPTrace')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.ylim(-(2*abs(dx1)-2*abs(dx1)/10),2*abs(dx1)/10)


    ax.set_title('Fault Plane View (HWB→FWB)')


    plt.arrow(0,0,v_1[0],-v_1[1],color='blue',head_length=1,head_width=2,label='nsv')
    ax.legend()

    st.pyplot(fig)



    #------------------------------------------------------------------
    c1=np.sin(sf)
    c2=np.cos(sf)
    c3=0
    #------------------------------------------------------------------
    t1=-a3/b3
    t2=-(a1*c1+a2*c2+a3*c3)/(b1*c1+b2*c2+b3*c3)
    #------------------------------------------------------------------
    r1=a+t1*b  #r1 is an array
    r2=a+t2*b  #r2 is an array
    #-------------------------------------------------------------------
    strike_sep=LA.norm(r1)
    dip_sep=LA.norm(r2)
    #-------------------------------------------------------------------
    s1v=np.array([np.sin(sf),np.cos(sf),0])
    ang=(np.arccos(np.clip(np.dot(s1v,r1), -1.0, 1.0)))*(180/np.pi)
    #---------------------------------------
    if ang==0:
        StrikeSeparationSense='Right_Separated'
    elif ang==180:
        StrikeSeparationSense='Left Separated'
    elif m.isnan(strike_sep):
        StrikeSeparationSense='No strike separation'
    #--------------------------------------------
    if r2[2]<0:
        DipSeparationSense='Down Separated'
    elif r2[2]>0:
        DipSeparationSense='Up Separated'
    else:
        DipSeparationSense='No Dip Separation'

    ans_array_sep=np.array([strike_sep,dip_sep])
    RO_ans_array_sep=np.round(ans_array_sep,3)


    st.write('Solution:')
    st.write('• StrikeSeparation=',RO_ans_array_sep[0],'   ,  ',StrikeSeparationSense)
    st.write('• DipSeparation=',RO_ans_array_sep[1],'   ,  ',DipSeparationSense)



######################################################################################################################################
#----------------------------------------------4 Calculation of Apparent thickness of Bed---------------------------------------------
######################################################################################################################################



elif dropdown=="4 Calculation of Apparent thickness of Bed":
    st.title("Calculation of Apparent thickness of Bed:")
    st.write("""
                ##### •Instructions: Enter strike value that is towards left of the dip direction 
            """)
    st.write("""
                ###### •Sample questions:
            """)
    st.write("""
                ###### 1. A bed of 'sandstone A' has attitude: 20°/68°→110° and orthogonal thcikness of 10 unit. A borehole is made which has 30° plunge towards 40°. What may the thickness of 'sandstone A' encountered in the borehole?
            """)
    st.write("""
                ###### 2. A bed of 'sandstone B' has attitude: 20°/68°→110° and orthogonal thcikness of 200 unit. A horizontal borehole is made towards 110°. What may the thickness of 'sandstone B' encountered in the borehole?
            """)


    st.write("""
                #### Provide Inputs:
                #### 
            """)
    
    
    # Create a 3-column layout
    col1, col2, col3 = st.columns(3)
# Add content to the first column
    with col1:
        bed_dip_deg=st.number_input('Enter dip of bed (in °):')
        bed_dip_dir_deg=st.number_input('Enter dip direction of bed (in °)')
# Add content to the second column
    with col3:
    
        ortho_thickness=st.number_input('Enter orthogonal thickness of bed(unit)')

# Add content to the third column
    with col2:
    
        bh_plunge_deg=st.number_input('Enter plunge of borehole (in °):')
        bh_trend_deg=st.number_input('Enter trend of borehole (in °):')


    bed_dip=np.deg2rad(bed_dip_deg)
    bed_dip_dir=np.deg2rad(bed_dip_dir_deg)
    bh_plunge=np.deg2rad(bh_plunge_deg)
    bh_trend=np.deg2rad(bh_trend_deg)

    if bed_dip_dir_deg <90:
        bed_strike_deg=bed_dip_dir_deg+270
    elif bed_dip_dir_deg >=90:
        bed_strike_deg=bed_dip_dir_deg-90

    bed_strike=np.deg2rad(bed_strike_deg)
    #-------------------------------------
    al=np.cos(bh_plunge)*np.sin(bh_trend)
    bl=np.cos(bh_plunge)*np.cos(bh_trend)
    cl=-np.sin(bh_plunge)
    #-------------------------------------
    an=np.sin(bed_dip)*np.cos(bed_strike)
    bn=-np.sin(bed_dip)*np.sin(bed_strike)
    cn=np.cos(bed_dip)
    #-------------------------------------
    m=-ortho_thickness/(al*an+bl*bn+cl*cn)
    Point=m*np.array([al,bl,cl])
    Dist=np.around(LA.norm(Point),3)
    #-------------------------------------
    st.write('Solution:')
    st.write('• Apparent Thickness of the bed=',Dist)




######################################################################################################################################
#----------------------------------------------5 Calculation of True Dip from 2 Apparent dip and dip directions-----------------------
######################################################################################################################################



elif dropdown=="5 Calculation of True Dip from 2 Apparent dip and dip directions":
    st.title("Calculation of True Dip from 2 Apparent dip and dip directions:")

    st.write("""
                ###### •Sample question:
            """)
    st.write("""
                ###### 1. Find the attitude of the contact between 2 uniformly planar beds where 2 apparent dips: 37°→N53°E & 44°→N26°E .
    
            """)
    st.write("""
                #### Provide Inputs:
                #### 
            """)
    

    AppDip1_deg=st.number_input('Enter apparent dip1 of bed (in degrees):')
    DipDir1_deg=st.number_input('Enter dip direction1 of bed (in degrees)')
    AppDip2_deg=st.number_input('Enter apparent dip2 of bed (in degrees):')
    DipDir2_deg=st.number_input('Enter dip direction2 of bed (in degrees)')


    AppDip1=np.deg2rad(AppDip1_deg)
    DipDir1=np.deg2rad(DipDir1_deg)
    AppDip2=np.deg2rad(AppDip2_deg)
    DipDir2=np.deg2rad(DipDir2_deg)
    #---------------------------------
    a1=np.cos(AppDip1)*np.sin(DipDir1)
    b1=np.cos(AppDip1)*np.cos(DipDir1)
    c1=-np.sin(AppDip1)
    #---------------------------------
    a2=np.cos(AppDip2)*np.sin(DipDir2)
    b2=np.cos(AppDip2)*np.cos(DipDir2)
    c2=-np.sin(AppDip2)
    #---------------------------------
    l1=[a1,b1,c1]
    l2=[a2,b2,c2]
    #---------------------------------
    v1=np.array(l1)
    v2=np.array(l2)
    #---------------------------------
    n=np.cross(v1,v2)
    #---------------------------------
    if n[2]>0:
        normal_vector=n
    elif n[2]<=0:
        normal_vector=-n
#---------------------------------
    dip_dir_vec=np.array([normal_vector[0],normal_vector[1]])
    nv=np.array([0,1])
#---------------------------------
    inner = np.inner(dip_dir_vec, nv)
    norms = LA.norm(dip_dir_vec) * LA.norm(nv)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg1 = np.rad2deg(rad)

    if dip_dir_vec[0]<0:
        DipDirection=360-deg1
    elif dip_dir_vec[0]>=0:
        DipDirection=deg1
#------------------------------------
    TrueDip=(np.arccos(abs(normal_vector[2])/LA.norm(normal_vector)))*(180/np.pi)
#------------------------------------
    if DipDirection<90:
        Strike=DipDirection+90
    elif DipDirection>=90 and DipDirection<270:
        Strike=DipDirection-90
    elif DipDirection>=270:
        Strike=DipDirection+90-360
    ans_array_truedip=np.array([Strike,TrueDip,DipDirection])
    RO_ans_array_truedip=np.round(ans_array_truedip,3)

    st.write('Solution:')
    st.write('• Attitude of the bed is:\n',RO_ans_array_truedip[0],'°/',RO_ans_array_truedip[1],'°→',RO_ans_array_truedip[2],'°')




######################################################################################################################################
#----------------------------------------------6 Orientation of intersection line of 2 planes-----------------------------------------
######################################################################################################################################




elif dropdown=="6 Orientation of intersection line of 2 planes":
    st.title("Orientation of intersection line of 2 planes:")
    st.write("""
                #### •Instructions: Always enter the strike which is towards left of the dip direction!
            """)
    st.write("""
                ###### •Sample question:
            """)
    st.write("""
                ###### 1. Plane1:12°/45°→E, Plane2:90°/30°→S. Find the attitude of the line of intersection of the planes.
    
            """)
    st.write("""
                #### Provide Inputs:
                #### 
            """)
    
    Strike1_deg=st.number_input('Enter strike of plane1 (in degrees):')
    DipDir1_deg=st.number_input('Enter dip of plane1 (in degrees):')
    Strike2_deg=st.number_input('Enter strike of plane2 (in degrees):')
    DipDir2_deg=st.number_input('Enter dip of plane2 (in degrees):')


    s1=np.deg2rad(Strike1_deg)
    d1=np.deg2rad(DipDir1_deg)
    s2=np.deg2rad(Strike2_deg)
    d2=np.deg2rad(DipDir2_deg)
    #--------------------------------- 
    a1=np.sin(d2)*np.sin(s2)*np.cos(d1)-np.sin(d1)*np.sin(s1)*np.cos(d2)
    b1=np.sin(d2)*np.cos(s2)*np.cos(d1)-np.sin(d1)*np.cos(s1)*np.cos(d2)
    c1=np.sin(d2)*np.cos(s2)*np.sin(d1)*np.sin(s1)-np.sin(d1)*np.cos(s1)*np.sin(d2)*np.sin(s2)
    #---------------------------------
    #direction vector of intersection line:
    
    if c1<=0:                                        #corrected!dipdirection
        il=np.array([a1,b1,c1])
    elif c1>0:
        il=np.array([-a1,-b1,-c1])

    nv=np.array([0,1,0])#north vector

    #-------------------------------------
    il_dir_vec=np.array([il[0],il[1],0])  #intersection line direction!
    #------------------------------------
    def unit_vector(vector):                                                  #
        return vector / np.linalg.norm(vector)                                #
    #------------------------------------
    il_dir_vec_u = unit_vector(il_dir_vec)                                                  #  Angle bet 2 vectors
    nv_u = unit_vector(nv)                                                    #
    deg1=(np.arccos(np.clip(np.dot(il_dir_vec_u,nv_u), -1.0, 1.0)))*(180/np.pi)      #

    if il_dir_vec[0]<0:
        Trend=360-deg1
    elif il_dir_vec[0]>=0:
        Trend=deg1
    #------------------------------------
    il_u = unit_vector(il) 
    Plunge=np.rad2deg(np.arccos(np.clip(np.dot(il_dir_vec_u, il_u), -1.0, 1.0)))


    st.write('Solution:')
    st.write('Attitude of the line of intersection:')
    st.write('• Trend=',np.round(Trend,3),'°')
    st.write('• Plunge=',np.round(Plunge,3),'°')






######################################################################################################################################
#--------------------------------------7 Finding orientation of plane passing through 2 lines-----------------------------------------
######################################################################################################################################



elif dropdown=="7 Finding orientation of plane passing through 2 lines":
    st.title("Finding orientation of plane passing through 2 lines:")


    st.write("""
                ###### •Sample question:
            """)
    
    st.write("""
                ###### 1. Line1: 40° → 110°, Line2: 30° → 160°. Find the attitude of the plane containing the 2 lines. 
            """)
    "[**Line1: x° → y° implies Line1 plunges x° towards y° ]"

    st.write("""
                #### Provide Inputs:
                #### 
            """)
    Plunge1_deg=st.number_input('Enter plunge of line1 (in degrees):')
    Trend1_deg=st.number_input('Enter trend of line1 (in degrees):')
    Plunge2_deg=st.number_input('Enter plunge of line2 (in degrees):')
    Trend2_deg=st.number_input('Enter trend of line2 (in degrees):')


    p1=np.deg2rad(Plunge1_deg)
    t1=np.deg2rad(Trend1_deg)
    p2=np.deg2rad(Plunge2_deg)
    t2=np.deg2rad(Trend2_deg)

    a=np.cos(p2)*np.cos(t2)*np.sin(p1)-np.cos(p1)*np.cos(t1)*np.sin(p2)
    b=np.cos(p1)*np.sin(t1)*np.sin(p2)-np.cos(p2)*np.sin(t2)*np.sin(p1)
    c=np.cos(p1)*np.sin(t1)*np.cos(p2)*np.cos(t2)-np.cos(p1)*np.cos(t1)*np.cos(p2)*np.sin(t2)

#---------------------------------
    normal_vector=np.array([a,b,c])
    d=LA.norm(normal_vector)

    TrueDip=(np.arccos(abs(c)/d))*(180/np.pi)


    if c>=0:                                        #corrected!dipdirection
        dip_dir_vec=np.array([a,b])
    elif c<0:
        dip_dir_vec=np.array([-a,-b])


    nv=np.array([0,1])

    inner = np.inner(dip_dir_vec, nv)
    norms = LA.norm(dip_dir_vec) * LA.norm(nv)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg1 = np.rad2deg(rad)

    if dip_dir_vec[0]<0:                 #logic for measuring angle >180
        DipDirection=360-deg1
    elif dip_dir_vec[0]>=0:
        DipDirection=deg1
    
    if DipDirection<90:
        Strike=DipDirection+90
    elif DipDirection>=90 and DipDirection<270:
        Strike=DipDirection-90
    elif DipDirection>=270:
        Strike=DipDirection+90-360
        
    ans_array=np.array([Strike,TrueDip,DipDirection])
    RO_ans_array=np.round(ans_array,3)

    st.write('Solution:')
    st.write('• Attitude of the bed is:\n',RO_ans_array[0],'°/',RO_ans_array[1],'°→',RO_ans_array[2],'°')







######################################################################################################################################
#----------------------------------------------8 Resultant Vector Calculator---------------------------------------------------------
######################################################################################################################################






if dropdown=="8 Resultant Vector Calculator":
    st.title("Resultant Vector Calculator :")
    st.write("""
                #### •Instructions: If vector is Upplunging then put (-1) and if it is downplunging or not-plunging put (+1)
            """)
    st.write("""
                ##### •Sample questions:
            """)

    st.write("""
                ###### 1. Calculate the magnitude and orientation of the resultant vector.
                ###### σ1 : Plunge= 40°,    Trend= 30°,  Magnitude= 5 units,     UpPlunging
                ###### σ2 : Plunge= 72°,    Trend= 66°,  Magnitude= 9 units,    DownPlunging
            """)
    
    st.write("""
                #
            """)

    
    st.write("""
                ###### 2. Calculate the magnitude and orientation of the resultant vector.
                ###### σ1 : Plunge= 0°,    Trend= 15°,  Magnitude= 5 units,     NonPlunging
                ###### σ2 : Plunge= 0°,    Trend= 50°,  Magnitude= 10 units,    NonPlunging
            """)
    
    st.write("""
                #### Provide Inputs:
                #### 
            """)
    
             
    # Create a 2-column layout
    col1, col2 = st.columns(2)
# Add content to the first column
    with col1:
        p_v1_deg=st.number_input('Plunge of vector1 (in °):')
        t_v1_deg=st.number_input('Trend of vector1 (in °):')
        m_v1=st.number_input('Magnitude of vector1 (unit):')
        UDsense_v1=st.slider('UpDownPlunge Sense of Vector1:',-1,+1,+1,step=1)
# Add content to the second column
    with col2:
    
        p_v2_deg=st.number_input('Plunge of vector2 (in °):')
        t_v2_deg=st.number_input('Trend of vector2 (in °):')
        m_v2=st.number_input('Magnitude of vector2 (unit):')
        UDsense_v2=st.slider('UpDownPlunge Sense of Vector2:',-1,+1,+1,step=1)

#----------------------------------------------------------------
    p_v1=np.deg2rad(p_v1_deg)
    t_v1=np.deg2rad(t_v1_deg)
    p_v2=np.deg2rad(p_v2_deg)
    t_v2=np.deg2rad(t_v2_deg)
#------------------------
    a1=UDsense_v1*m_v1*(np.cos(p_v1)*np.sin(t_v1))
    b1=UDsense_v1*m_v1*(np.cos(p_v1)*np.cos(t_v1))
    c1=UDsense_v1*m_v1*(-np.sin(p_v1))
    v1=np.array([a1,b1,c1])
    #------------------------
    a2=UDsense_v2*m_v2*(np.cos(p_v2)*np.sin(t_v2))
    b2=UDsense_v2*m_v2*(np.cos(p_v2)*np.cos(t_v2))
    c2=UDsense_v2*m_v2*(-np.sin(p_v2))
    v2=np.array([a2,b2,c2])
    #------------------------
    R=v1+v2
    M=LA.norm(R)    #magnitude
#------------------------
    if R[2]>0:
        Down_Plunge_R=-R
    elif R[2]<=0:
        Down_Plunge_R=R 
    #------------------------
    Trend_Vector=np.array([Down_Plunge_R[0],Down_Plunge_R[1],0])
    nv=np.array([0,1,0])
    #------------------------
    def unit_vector(vector):                                                           #
            return vector / np.linalg.norm(vector)                                         #
    #------------------------------------
    Trend_Vector_u = unit_vector(Trend_Vector)                                         #  Angle bet 2 vectors
    nv_u = unit_vector(nv)                                                             #
    Trend=(np.arccos(np.clip(np.dot(Trend_Vector_u,nv_u), -1.0, 1.0)))*(180/np.pi)     #

    if Trend_Vector[0]<0:
        Trend=360-Trend
    elif Trend_Vector[0]>=0:
        Trend=Trend
    #-----------------------------------------------------
    U=[R[0],R[1],0]
    #-----------------------------------------------------
    R_u = unit_vector(R)                                         
    U_u = unit_vector(U)
    Plunge=(np.arccos(np.clip(np.dot(R_u,U_u), -1.0, 1.0)))*(180/np.pi) 

    if R[2]>0:
        UD='UP Plunging'
    elif R[2]<0:
        UD='Down Plunging'
    elif R[2]==0:
        UD='No Plunge'
    
    ans_arrsay=np.array([Trend,Plunge,M])
    RO_ans_array=np.round(ans_array,3)

    st.write('Solution:')
    st.write('• Orientation of the resultant vector:',RO_ans_array[1],'°→',RO_ans_array[0],'°')
    st.write('• Magnitude=',RO_ans_array[2])
    st.write('\n• UpDownSense=',UD)




######################################################################################################################################
#-------------------------------------------------------------The End-----------------------------------------------------------------
######################################################################################################################################