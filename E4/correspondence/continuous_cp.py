"""Continuous local closest point on an empirical STSP residual-zero sheet.

No true curve, parameter, normal, Oracle point or candidate library is accepted.
The scalar chart uses a tangent axis from empirical h2 PCA. Every objective
and derivative evaluation solves the actual STSP equations, not a spline or NN.
"""
import time
import numpy as np
from scipy.optimize import brentq


def continuous_cp(operator, reference, solve_fiber, controls, grid_nodes=41,
                  validate_normal_starts=True,stationarity_tolerance=1e-10):
    start=time.perf_counter();reference=np.asarray(reference,dtype=float)
    evaluations=0;root_evaluations=0;cache={};all_roots=[]
    try:
        initial=operator.evaluate(reference)
        B=initial.eigenvectors[:,:2].copy();T=initial.eigenvectors[:,2].copy()
        if not np.allclose(B.T@T,0,atol=1e-12):raise ArithmeticError('Nonorthogonal chart')
        def chart(s, derivatives=True):
            nonlocal evaluations,root_evaluations
            s=float(s)
            if s in cache:return cache[s]
            evaluations+=1
            fit=solve_fiber(operator,reference+s*T,B,operator.h,controls=controls)
            root_evaluations+=int(fit.diagnostics.get('function_evaluations',0))
            if fit.status!='success':raise ArithmeticError('chart_root:'+fit.status)
            z=fit.point;state=operator.evaluate(z)
            J=np.column_stack([operator.directional(state,B[:,k])['DR'] for k in range(2)])
            v=operator.directional(state,T)['DR']
            A=B.T@J
            condition=float(np.linalg.cond(A))
            if condition>controls['jacobian_condition_max']:raise ArithmeticError('chart_Jacobian_condition')
            tangent=T+B@np.linalg.solve(A,-B.T@v)
            grad=float((z-reference)@tangent)
            out=dict(s=s,point=z,gradient=grad,objective=float(np.sum((z-reference)**2)),
                projected_residual=float(np.linalg.norm(B.T@state.residual)),
                full_residual=float(np.linalg.norm(state.residual)),
                tangent=tangent,condition=condition,root=fit.diagnostics)
            cache[s]=out
            return out
        central=chart(0.)
        # Any point with |T'(z-reference)| > incumbent distance cannot improve
        # the incumbent. A valid single-sheet chart therefore reduces the
        # continuous nearest-point search to this finite interval.
        radius=np.sqrt(central['objective'])*(1+1e-10)+1e-12
        grid=np.linspace(-radius,radius,int(grid_nodes))
        samples=[chart(s) for s in grid]
        actual_grid_nodes=int(grid_nodes)
        # Adaptive construction is part of the declared scalar optimizer.
        # A nonmonotone gradient receives a denser census; this is not a
        # retry of a failed STSP fit or a change to its acceptance criteria.
        for refinement in [41,81]:
            if np.all(np.diff([p['gradient'] for p in samples])>0):break
            if refinement<=actual_grid_nodes:continue
            actual_grid_nodes=refinement
            grid=np.linspace(-radius,radius,refinement)
            samples=[chart(s) for s in grid]
        # The central root provides a distance upper bound, not automatically
        # a CP candidate. At small noise, primal root roundoff can dominate
        # differences between nearly identical squared distances.
        candidates=[samples[0],samples[-1]]
        if abs(central['gradient'])/np.linalg.norm(central['tangent'])<=stationarity_tolerance:
            candidates.append(central)
        for left,right in zip(samples[:-1],samples[1:]):
            if left['gradient']==0:
                candidates.append(left);all_roots.append(left['s'])
            if left['gradient']<0<right['gradient']:
                s=brentq(lambda value:chart(value)['gradient'],left['s'],right['s'],xtol=1e-13,rtol=1e-13)
                candidates.append(chart(s));all_roots.append(float(s))
        best=min(candidates,key=lambda c:c['objective'])
        # An endpoint or a nonstationary fallback is not silently accepted as CP.
        normalized_gradient=abs(best['gradient'])/np.linalg.norm(best['tangent'])
        if normalized_gradient>stationarity_tolerance:raise ArithmeticError('CP_stationarity_not_resolved')
        if abs(best['s'])>=radius*(1-1e-8):raise ArithmeticError('CP_at_chart_boundary')
        if best['projected_residual']>controls['projected_residual_tolerance'] or best['full_residual']>controls['full_residual_tolerance']:
            raise ArithmeticError('CP_final_STSP_residual')
        normal_checks=[]
        if validate_normal_starts:
            for k in range(2):
                for sign in [-1,1]:
                    origin=reference+best['s']*T+sign*.5*radius*B[:,k]
                    fit=solve_fiber(operator,origin,B,operator.h,controls=controls)
                    root_evaluations+=int(fit.diagnostics.get('function_evaluations',0))
                    if fit.status!='success':raise ArithmeticError('CP_normal_multistart:'+fit.status)
                    difference=float(np.linalg.norm(fit.point-best['point']))
                    normal_checks.append(difference)
                    if difference>2e-8:raise ArithmeticError('multiple_empirical_normal_sheet_roots')
        return dict(status='success',point=best['point'],seconds=time.perf_counter()-start,
            chart_origin=reference,chart_normal_basis=B,chart_tangent=T,
            domain=[-radius,radius],grid_nodes=actual_grid_nodes,initial_grid_nodes=int(grid_nodes),selected_s=best['s'],
            candidate_stationary_s=all_roots,objective=best['objective'],
            normalized_stationarity=normalized_gradient,projected_residual=best['projected_residual'],
            stationarity_tolerance=stationarity_tolerance,
            full_residual=best['full_residual'],chart_evaluations=evaluations,
            STSP_root_function_evaluations=root_evaluations,
            normal_multistart_point_differences=normal_checks,
            nearest_grid_s_difference=float(np.min(np.abs(grid-best['s']))),
            global_scope='Continuous minimum on the numerically verified local fitted sheet; outside the incumbent tangent interval excluded by Euclidean lower bound. This is not a formal global certificate for every spurious empirical residual component.',
            algorithm_inputs='Empirical STSP operator and reference point only; no true geometry')
    except Exception as error:
        return dict(status='failed',reason=type(error).__name__+': '+str(error),
            point=np.full(3,np.nan),seconds=time.perf_counter()-start,
            chart_evaluations=evaluations,STSP_root_function_evaluations=root_evaluations)
