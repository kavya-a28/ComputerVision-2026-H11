import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import graycomatrix,graycoprops
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
image=data.camera()
print("Image shape:",image.shape)
print("Image data type:",image.dtype)
grass_patches=[("Grass 1",250,270,400,420),("Grass 2",300,320,400,420),("Grass 3",350,370,400,420),("Grass 4",400,420,400,420)]
sky_patches=[("Sky 1",40,60,40,60),("Sky 2",40,60,150,170),("Sky 3",40,60,270,290),("Sky 4",40,60,400,420)]
patches=grass_patches+sky_patches
fig,ax=plt.subplots(figsize=(8,6))
ax.imshow(image,cmap="gray")
for name,y1,y2,x1,x2 in grass_patches:
    rect=Rectangle((x1,y1),x2-x1,y2-y1,fill=True,edgecolor="green",linewidth=3)
    ax.add_patch(rect)
for name,y1,y2,x1,x2 in sky_patches:
    rect=Rectangle((x1,y1),x2-x1,y2-y1,fill=True,edgecolor="blue",linewidth=3)
    ax.add_patch(rect)
ax.set_title("Original Image")
ax.axis("off")
plt.show()
patch_images={}
for name,y1,y2,x1,x2 in patches:
    patch_images[name]=image[y1:y2,x1:x2]
fig,axes=plt.subplots(2,4,figsize=(14,6))
for i,(name,patch) in enumerate([(x,patch_images[x]) for x in [p[0] for p in grass_patches]]):
    axes[0,i].imshow(patch,cmap="gray")
    axes[0,i].set_title(name)
    axes[0,i].axis("off")
for i,(name,patch) in enumerate([(x,patch_images[x]) for x in [p[0] for p in sky_patches]]):
    axes[1,i].imshow(patch,cmap="gray")
    axes[1,i].set_title(name)
    axes[1,i].axis("off")
plt.tight_layout()
plt.show()
results=[]
distances=[1]
angles=[0,np.pi/4,np.pi/2,3*np.pi/4]
for name,patch in patch_images.items():
    glcm=graycomatrix(patch,distances=distances,angles=angles,levels=256,symmetric=True,normed=True)
    contrast=graycoprops(glcm,"contrast").mean()
    dissimilarity=graycoprops(glcm,"dissimilarity").mean()
    homogeneity=graycoprops(glcm,"homogeneity").mean()
    energy=graycoprops(glcm,"energy").mean()
    correlation=graycoprops(glcm,"correlation").mean()
    results.append([name,contrast,dissimilarity,homogeneity,energy,correlation])
print("\nGLCM Texture Features")
print(f"{'Patch':<12}{'Contrast':<15}{'Dissimilarity':<18}{'Homogeneity':<15}{'Energy':<12}{'Correlation':<15}")
for row in results:
    print(f"{row[0]:<12}{row[1]:<15.4f}{row[2]:<18.4f}{row[3]:<15.4f}{row[4]:<12.4f}{row[5]:<15.4f}")
grass_results=[r for r in results if "Grass" in r[0]]
sky_results=[r for r in results if "Sky" in r[0]]
plt.figure(figsize=(8,6))
plt.scatter([r[2] for r in grass_results],[r[5] for r in grass_results],color="green",s=100,label="Grass")
plt.scatter([r[2] for r in sky_results],[r[5] for r in sky_results],color="blue",s=100,label="Sky")
for r in grass_results:
    plt.annotate(r[0],(r[2],r[5]),xytext=(5,5),textcoords="offset points")
for r in sky_results:
    plt.annotate(r[0],(r[2],r[5]),xytext=(5,5),textcoords="offset points")
plt.xlabel("GLCM Dissimilarity")
plt.ylabel("GLCM Correlation")
plt.title("GLCM Dissimilarity vs Correlation")
plt.legend()
plt.grid(True)
plt.show()
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection="3d")
ax.scatter([r[2] for r in grass_results],[r[1] for r in grass_results],[r[5] for r in grass_results],color="green",s=100,label="Grass")
ax.scatter([r[2] for r in sky_results],[r[1] for r in sky_results],[r[5] for r in sky_results],color="blue",s=100,label="Sky")
for r in results:
    ax.text(r[2],r[1],r[5],r[0])
ax.set_xlabel("Dissimilarity")
ax.set_ylabel("Contrast")
ax.set_zlabel("Correlation")
ax.set_title("3D GLCM Feature Space")
ax.legend()
plt.show()