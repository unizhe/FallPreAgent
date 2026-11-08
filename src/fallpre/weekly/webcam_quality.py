"""Fixed-camera checks with bounded, independently verified startup recovery."""
import cv2
import numpy as np


class LiveCameraQuality:
    """Only an unverified startup reference may be replaced automatically.

    Reacquisition checks a fixed candidate over multiple real frames and source
    time. A trusted reference or confirmed camera movement is never overwritten.
    No pixels are written to disk or exposed in diagnostic exports.
    """
    def __init__(self):
        self.reference = None
        self.reference_boxes = []
        self.shape = None
        self.moved = False
        self.trusted = False
        self.pending = None
        self.last_t = None
        self.reference_t = None
        self.generation = 0
        self.motion_pending = None
        self.motion_recovery = None
        self.orb = cv2.ORB_create(nfeatures=800, edgeThreshold=12, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    @staticmethod
    def result(stable, code, reason, **metrics):
        return dict(camera_stable=stable, code=code, reason=reason, metrics=metrics)

    def _features(self, gray, boxes):
        mask = np.full(gray.shape, 255, np.uint8)
        for box in boxes:
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            x1,y1,x2,y2 = np.rint(box).astype(int)
            cv2.rectangle(mask, (x1-8,y1-8), (x2+8,y2+8), 0, -1)
        values = gray[mask>0]
        metrics = dict(background_pixels=int(values.size))
        if values.size < 256:
            return None, None, self.result(None,'background_hidden','可见背景不足，暂不能核对相机稳定性',**metrics)
        if np.mean(values<25)>.8 or np.mean(values>245)>.8:
            return None, None, self.result(None,'lighting_unreliable','背景过暗或过曝，暂不能核对相机稳定性',**metrics)
        keys,desc = self.orb.detectAndCompute(gray,mask)
        metrics['current_features'] = 0 if desc is None else len(desc)
        if desc is None or len(desc)<12:
            return None,None,self.result(None,'background_features_missing','背景纹理不足，暂不能核对相机稳定性',**metrics)
        return keys,desc,None

    def _compare(self, reference, ref_boxes, gray, boxes):
        excluded=ref_boxes+boxes
        current_keys,current_desc,error=self._features(gray,excluded)
        if error:return error
        ref_keys,ref_desc,error=self._features(reference,excluded)
        if error:return self.result(None,'reference_hidden','起始背景目前无法匹配，等待可见背景',**error['metrics'])
        pairs=self.matcher.knnMatch(ref_desc,current_desc,k=2)
        matches=[p[0] for p in pairs if len(p)==2 and p[0].distance<.72*p[1].distance]
        metrics=dict(current_features=len(current_desc),reference_features=len(ref_desc),matches=len(matches))
        if len(matches)<12:
            return self.result(None,'background_matches_missing','背景匹配不足，暂不能确认画面稳定',**metrics)
        origin=np.float32([ref_keys[m.queryIdx].pt for m in matches])
        current=np.float32([current_keys[m.trainIdx].pt for m in matches])
        transform,inliers=cv2.estimateAffinePartial2D(origin,current,method=cv2.RANSAC,ransacReprojThreshold=2.)
        metrics.update(inliers=0 if inliers is None else int(inliers.sum()),
                       inlier_fraction=0. if inliers is None else round(float(inliers.mean()),3))
        if transform is None or inliers is None or inliers.sum()<12 or inliers.mean()<.6:
            return self.result(None,'background_geometry_unknown','背景几何不一致，暂不能确认画面稳定',**metrics)
        selected=origin[inliers.ravel().astype(bool)]
        coverage=float(np.prod(np.ptp(selected,axis=0))/gray.size)
        metrics['coverage']=round(coverage,4)
        if coverage<.06:return self.result(None,'background_coverage_low','背景特征过于集中，暂不能确认相机稳定',**metrics)
        h,w=gray.shape
        grid=np.array([[0,0],[w,0],[0,h],[w,h],[w/2,h/2]],np.float32)
        projected=np.c_[grid,np.ones(len(grid))]@transform.T
        shift=float(np.max(np.linalg.norm(projected-grid,axis=1)))
        metrics.update(background_shift_px=round(shift,3),reference_width=w)
        if shift>4.:return self.result(False,'camera_moved','检测到相机移动，请固定相机后重新启动测试',**metrics)
        return self.result(True,'stable','可见背景与本次参考画面一致',**metrics)

    def update(self,image,persons,t=None):
        t=float(t if t is not None else (0. if self.last_t is None else self.last_t+.1))
        if not np.isfinite(t) or t<0 or self.last_t is not None and t<=self.last_t:
            raise ValueError('背景检查时间戳必须递增')
        if self.last_t is not None and t-self.last_t>.35:
            self.pending=None
            self.motion_pending=None
            self.motion_recovery=None
        self.last_t=t
        height,width=image.shape[:2]
        if self.shape is not None and self.shape!=(height,width):self.moved=True
        self.shape=(height,width)
        def finish(result):
            result['metrics'].update(reference_generation=self.generation,reference_time_s=self.reference_t,
                                     reference_trusted=self.trusted,reacquisition_count=0 if self.pending is None else self.pending['count'])
            return result
        if self.moved:return finish(self.result(False,'camera_moved','画面位置或尺寸已变化，请固定相机后重新启动测试'))
        scale=min(1.,480/width)
        gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(round(width*scale),round(height*scale)))
        boxes=[np.asarray(p['box'],dtype=float)*scale for p in persons]
        if self.reference is None:
            _,_,error=self._features(gray,boxes)
            if error:return finish(error)
            self.reference,self.reference_boxes,self.reference_t=gray.copy(),boxes,t
            self.generation+=1
            return finish(self.result(None,'reference_warmup','已建立本次背景参考，等待下一帧核对'))
        result=self._compare(self.reference,self.reference_boxes,gray,boxes)
        if result['camera_stable'] is False:
            # A single affine-fit outlier must stop current inference but must
            # not permanently poison a trusted reference. Confirm on real,
            # consecutive source observations, never by rebasing the image.
            self.motion_recovery=None
            if self.motion_pending is None:
                self.motion_pending=dict(since=t,count=1)
            else:
                self.motion_pending['count']+=1
            if self.motion_pending['count']>=3 and t-self.motion_pending['since']>=.2-1e-8:
                self.moved=True
                return finish(result)
            result.update(code='camera_motion_pending',reason='背景出现位移证据，暂停判断并等待连续核实')
            return finish(result)
        if result['camera_stable'] is True:
            if self.motion_pending is not None or self.motion_recovery is not None:
                self.motion_pending=None
                if self.motion_recovery is None:
                    self.motion_recovery=dict(since=t,count=1)
                else:
                    self.motion_recovery['count']+=1
                if self.motion_recovery['count']<2 or t-self.motion_recovery['since']<.1-1e-8:
                    result.update(camera_stable=None,code='camera_motion_recheck',
                                  reason='画面重新匹配原背景，等待第二次验证')
                    return finish(result)
                self.motion_recovery=None
            self.trusted=True
            self.pending=None
            return finish(result)
        self.motion_pending=None
        self.motion_recovery=None
        if self.trusted:return finish(result)
        # Do not declare a new scene stable from a single replacement frame.
        _,_,error=self._features(gray,boxes)
        if error:
            self.pending=None
            return finish(result)
        candidate_result=None
        if self.pending is not None:
            candidate_result=self._compare(self.pending['image'],self.pending['boxes'],gray,boxes)
        if candidate_result is None or candidate_result['camera_stable'] is not True:
            self.pending=dict(image=gray.copy(),boxes=boxes,since=t,count=1)
            return finish(result)
        self.pending['count']+=1
        if self.pending['count']>=4 and t-self.pending['since']>=.35-1e-8:
            self.reference=self.pending['image']
            self.reference_boxes=self.pending['boxes']
            self.reference_t=self.pending['since']
            self.generation+=1
            self.trusted=True
            self.pending=None
            candidate_result.update(code='reference_recovered',reason='启动参考匹配失败；新参考已连续验证，恢复背景检查')
            return finish(candidate_result)
        return finish(result)
