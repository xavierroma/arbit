```mermaid
flowchart LR
  %% Threads
  A[Camera / Sensor I/O\nthread]:::thr
  B[Frontend Tracker\n]:::thr
  C[Place Recognition\nDBoW / matcher]:::thr
  D[Backend Local Mapping\ntriangulate + local BA]:::thr
  E[Map Store / State\nshared, versioned]:::store
  F[Renderer / AR Output\nthread]:::thr

  %% Queues
  q1((q: FrameIn)):::q
  q2((q: KF_Candidate)):::q
  q3((q: KF_Accepted)):::q
  q4((q: Reloc_Request)):::q
  q5((q: Reloc_Result)):::q
  q6((q: Map_Update)):::q

  %% Flow
  A --> q1
  q1 --> B

  B -- if tracking OK --> F
  B -- pose-only BA uses --> E

  B -- "KeyframeCandidate" --> q2
  q2 --> C
  q2 --> D

  C -- "PlaceHits / LoopCandidates" --> D

  D -- "KeyframeAccepted + TriangulatedPoints" --> q3
  q3 --> E

  D -- "Local BA & Culling" --> q6
  q6 --> E

  E -- "read-only snapshots" --> B
  E -- "read-only snapshots" --> F

  %% Relocalization path
  B -- "Reloc_Request" --> q4
  q4 --> C
  C -- "Reloc_Result (PnP pose + matches)" --> q5
  q5 --> B

  classDef thr fill:#111,color:#fff,stroke:#555,rx:8,ry:8
  classDef q fill:#1a2a,stroke:#355,rx:25,ry:25,color:#bdf
  classDef store fill:#131a26,stroke:#355,color:#fff
  ```

  ```mermaid
  sequenceDiagram
  participant Cam as Camera I/O
  participant FE as Frontend Tracker
  participant PR as Place Recognition
  participant LM as Local Mapping
  participant MS as Map Store
  participant AR as Renderer

  Cam->>FE: FrameIn{img,pyr,ts,K,dist,exposure}
  FE->>FE: Track LK (confirmed & tentative)
  FE->>FE: Pose-only BA vs local map
  FE->>AR: PoseUpdate{T_wc, tracking_quality}
  alt Need new keyframe?
    FE->>PR: KF_Candidate{KFn_meta,kps,desc,obs}
    FE->>LM: KF_Candidate{...}
    PR-->>LM: PlaceHits{candidate_KFs, match_hints}
    LM->>LM: Select neighbors (covisibility/PR hints)
    LM->>LM: Match descriptors (KF_n ↔ neighbors)
    LM->>LM: Triangulate_with_neighbors (inlier gating)
    LM->>LM: Local BA (window KFs + points)
    LM->>MS: KeyframeAccepted + MapDelta
    MS-->>FE: MapSnapshot v{n+1}
    MS-->>AR: MapSnapshot v{n+1}
  end
  ```