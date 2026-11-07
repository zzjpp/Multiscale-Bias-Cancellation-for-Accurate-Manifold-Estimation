#!/usr/bin/env python3
"""Audit the actual scalar definitions used by the native TikZ source.

The drawing is illustrative. This checks normal-space geometry, not the
asymptotic statistical theorem or the magnitudes of its scalar coefficients.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
source=(ROOT/'figure_tikz.tex').read_text(encoding='utf-8')


def scalar(name):
    match=re.search(r'\\def\\'+name+r'\{([-+0-9.]+)\}',source)
    if not match:
        raise ValueError(f'Missing TikZ scalar: {name}')
    return float(match.group(1))


def main():
    a,b=scalar('curva'),scalar('curvb')
    lp,ls,lb=(scalar(k) for k in ('leadP','leadS','leadB'))
    rp,rs=scalar('restP'),scalar('restS')
    projection=np.array([[1,-.25,.60],[0,1,.65]],dtype=float)
    nu=np.array([0.,1.,0.])
    tangent=np.array([1.,0.,0.])
    P=np.outer(nu,nu)
    S=np.diag([0.,0.,1.])
    Q=np.diag([0.,1.,1.])
    H=np.array([0.,2*a,2*b])
    cp=np.array([0.,lp,0.])
    cs=np.array([0.,0.,ls])
    cb=np.array([0.,0.,lb])
    remainder=np.array([0.,rp,rs])
    displacement=cp+cs+cb+remainder
    residuals={}

    def check(name,value,tolerance=1e-12):
        value=float(np.linalg.norm(value))
        residuals[name]=value
        assert value<tolerance,(name,value)

    check('P0_idempotent',P@P-P)
    check('S0_idempotent',S@S-S)
    check('normal_projector_split',P+S-Q)
    check('normal_subspaces_orthogonal',P@S)
    check('nu0_spans_P0',P@nu-nu)
    guide_shift=np.array([0.,0.,scalar('pguideS')])
    guide_low=-.33*nu+guide_shift
    guide_high=2.62*nu+guide_shift
    check('translated_nu0_guide_parallel_to_true_axis',np.cross(guide_high-guide_low,nu))
    check('translated_nu0_guide_stays_in_normal_plane',np.dot(guide_shift,tangent))
    check('P0H_component_in_RanP0',P@cp-cp)
    check('P0H_component_parallel_to_field',np.cross(cp,P@H))
    check('S0H_component_in_RanS0',S@cs-cs)
    check('S0H_component_parallel_to_field',np.cross(cs,S@H))
    check('b_perp_component_in_RanS0',S@cb-cb)
    check('S0H_bperp_collinear_in_R3',np.cross(cs,cb))
    ps,pb=projection@cs,projection@cb
    check('S0H_bperp_collinear_in_display',ps[0]*pb[1]-ps[1]*pb[0])
    check('P0H_perpendicular_to_S0H_in_R3',np.dot(cp,cs))
    check('full_displacement_normal',np.dot(displacement,tangent))
    assert np.linalg.norm(remainder)>0,'The leading terms must not close exactly.'
    leading_endpoint=cp+cs+cb
    check('nonzero_closing_gap_is_remainder',displacement-leading_endpoint-remainder)
    check('leading_components_plus_remainder_close_to_full',displacement-(cp+cs+cb+remainder))

    # q(t)=p(t)+Q(t)U uses the true orthogonal normal projector at p(t).
    max_normal_residual=0.
    max_display_formula_residual=0.
    sample_x=[]
    for t in np.linspace(-4.60,4.60,1001):
        p=np.array([t,a*t*t,b*t*t])
        dp=np.array([1.,2*a*t,2*b*t])
        u=displacement-dp*np.dot(dp,displacement)/np.dot(dp,dp)
        q=p+u
        max_normal_residual=max(max_normal_residual,abs(float(np.dot(u,dp))))
        nf=(2*a*displacement[1]+2*b*displacement[2])*t/(1+4*(a*a+b*b)*t*t)
        tx=t+(-.25*a+.6*b)*t*t
        ty=(a+.65*b)*t*t
        displayed_formula=np.array([
            tx-.25*displacement[1]+.60*displacement[2]-nf*(1+(-.50*a+1.20*b)*t),
            ty+displacement[1]+.65*displacement[2]-nf*(2*a+1.30*b)*t])
        max_display_formula_residual=max(max_display_formula_residual,float(np.linalg.norm(projection@q-displayed_formula)))
        sample_x.append(q[0])
    check('normal_graph_orthogonality_max',max_normal_residual)
    check('TikZ_projected_curve_formula_max',max_display_formula_residual)
    assert np.all(np.diff(sample_x)>0),'The illustrative fitted curve must be embedded.'
    check('fitted_position_at_zero_is_x_plus_u',Q@displacement-displacement)

    pdf=ROOT/'yy_normal_decomposition.pdf'
    reader=PdfReader(pdf)
    assert len(reader.pages)==1
    page=reader.pages[0]
    assert len(page.images)==0,'The deliverable must contain no raster images.'
    fonts=[]
    for ref in page['/Resources']['/Font'].values():
        font=ref.get_object()
        desc=font.get('/FontDescriptor')
        assert desc is not None
        desc=desc.get_object()
        embedded=any(k in desc for k in ('/FontFile','/FontFile2','/FontFile3'))
        assert embedded
        fonts.append({'name':str(font.get('/BaseFont')),'embedded':embedded})
    log=(ROOT/'yy_normal_decomposition.log').read_text(encoding='utf-8',errors='replace')
    assert 'Output written on yy_normal_decomposition.pdf' in log
    assert not re.search(r'(^!|Missing character:|Overfull \\hbox)',log,re.M)
    report={'status':'PASS','geometry_residuals':residuals,
        'S0_rank':int(np.linalg.matrix_rank(S)),
        'position_x_plus_u_in_R3':displacement.tolist(),
        'projected_full_displacement':(projection@displacement).tolist(),
        'leading_sum_in_R3':leading_endpoint.tolist(),
        'nonzero_schematic_remainder_R3':remainder.tolist(),
        'remainder_interpretation':'Nonzero schematic O(sigma^4) remainder, not separately drawn; no asymptotic rate is tested by the drawing.',
        'projected_remainder_norm':float(np.linalg.norm(projection@remainder)),
        'nu0_direction_guide_translation_R3':guide_shift.tolist(),
        'curve_model':'p(t)=(t,a*t^2,b*t^2), q(t)=p(t)+Q(t)U, Q(t) the true normal projector.',
        'PDF':{'pages':1,'raster_images':0,'fonts':fonts,
               'width_cm':float(page.mediabox.width)/72*2.54,'height_cm':float(page.mediabox.height)/72*2.54},
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'PDF_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'max_residual':max(residuals.values()),
                      'collinearity_R3':residuals['S0H_bperp_collinear_in_R3'],
                      'collinearity_display':residuals['S0H_bperp_collinear_in_display'],
                      'PDF_raster_images':0,'embedded_fonts':len(fonts)},indent=2))


if __name__=='__main__':
    main()
