import vtk, numpy as np, sys
sys.path.insert(0,'../bronchotrack')
from vtk.util.numpy_support import vtk_to_numpy

class Renderer:
    def __init__(self, mesh_path, size=160, fov=90, atten=0.002):
        r=vtk.vtkPolyDataReader(); r.SetFileName(mesh_path); r.Update()
        n=vtk.vtkPolyDataNormals(); n.SetInputData(r.GetOutput()); n.ConsistencyOn(); n.AutoOrientNormalsOff(); n.Update()
        self.ren=vtk.vtkRenderer(); self.rw=vtk.vtkRenderWindow(); self.rw.SetOffScreenRendering(1)
        self.rw.AddRenderer(self.ren); self.rw.SetSize(size,size); self.size=size
        m=vtk.vtkPolyDataMapper(); m.SetInputData(n.GetOutput()); m.ScalarVisibilityOff()
        a=vtk.vtkActor(); a.SetMapper(m); p=a.GetProperty(); p.SetColor(0.93,0.45,0.55); p.SetAmbient(0.0); p.SetDiffuse(1.0); p.SetSpecular(0.2)
        p.BackfaceCullingOff(); self.ren.AddActor(a)
        self.ren.RemoveAllLights(); self.light=vtk.vtkLight(); self.light.SetLightTypeToHeadlight(); self.light.SetPositional(True)
        self.light.SetConeAngle(179); self.light.SetAttenuationValues(1,0,atten); self.ren.AddLight(self.light)
        self.ren.AutomaticLightCreationOff(); self.ren.TwoSidedLightingOn()
        self.cam=self.ren.GetActiveCamera(); self.cam.SetViewAngle(fov)
        self.w2i=vtk.vtkWindowToImageFilter(); self.w2i.SetInput(self.rw); self.w2i.ReadFrontBufferOff()
    def render(self, pos, fwd, up):
        self.cam.SetPosition(*pos); self.cam.SetFocalPoint(*(pos+fwd*10)); self.cam.SetViewUp(*up)
        self.cam.SetClippingRange(0.3, 400); self.rw.Render()
        self.w2i.Modified(); self.w2i.Update()
        im=vtk_to_numpy(self.w2i.GetOutput().GetPointData().GetScalars()).reshape(self.size,self.size,-1)[::-1,:,:3]
        return np.ascontiguousarray(im[:,:,::-1])
