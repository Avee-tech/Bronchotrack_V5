"""Voxelise a closed airway surface mesh (VTK/STL/OBJ) into a binary mask (.npz with spacing/origin)."""
import sys, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy

def read(path):
    r = vtk.vtkSTLReader() if path.endswith(".stl") else vtk.vtkOBJReader() if path.endswith(".obj") else vtk.vtkPolyDataReader()
    r.SetFileName(path); r.Update(); return r.GetOutput()

def voxelize(pd, spacing=0.5, pad=3.0):
    b = np.array(pd.GetBounds()).reshape(3, 2)
    origin = b[:, 0] - pad
    dims = np.ceil((b[:, 1] + pad - origin) / spacing).astype(int) + 1
    img = vtk.vtkImageData(); img.SetSpacing([spacing] * 3); img.SetOrigin(origin.tolist())
    img.SetDimensions(dims.tolist()); img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    img.GetPointData().GetScalars().Fill(1)
    st = vtk.vtkPolyDataToImageStencil(); st.SetInputData(pd)
    st.SetOutputOrigin(origin.tolist()); st.SetOutputSpacing([spacing] * 3); st.SetOutputWholeExtent(img.GetExtent()); st.Update()
    cut = vtk.vtkImageStencil(); cut.SetInputData(img); cut.SetStencilConnection(st.GetOutputPort())
    cut.ReverseStencilOff(); cut.SetBackgroundValue(0); cut.Update()
    arr = vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars()).reshape(dims[::-1]).transpose(2, 1, 0)
    return arr > 0, origin

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    sp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    pd = read(src)
    clean = vtk.vtkCleanPolyData(); clean.SetInputData(pd); clean.Update()
    m, org = voxelize(clean.GetOutput(), sp)
    np.savez_compressed(out, mask=m, spacing=np.array([sp] * 3), origin=org)
    print("mask", m.shape, "filled", int(m.sum()), "vol mm3", m.sum() * sp ** 3)
