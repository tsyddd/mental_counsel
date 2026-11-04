package main

import (
    "bytes"
    "encoding/json"
    "io"
    "net/http"
    "net/url"
)

// All user scopes are derived from the authenticated cookie. The internal API
// is not directly published. No caller-supplied user ID is forwarded.
func (s *server) memoryForward(w http.ResponseWriter, r *http.Request, method, path string, body []byte) {
    req, err := http.NewRequestWithContext(r.Context(),method,s.media+path,bytes.NewReader(body))
    if err != nil { fail(w,500,"invalid memory request");return }
    req.Header.Set("Content-Type","application/json")
    resp,err:=s.client.Do(req)
    if err!=nil { fail(w,502,"memory service unavailable");return }
    defer resp.Body.Close()
    w.Header().Set("Content-Type",resp.Header.Get("Content-Type"))
    w.Header().Set("Cache-Control","no-cache")
    w.Header().Set("X-Accel-Buffering","no")
    w.WriteHeader(resp.StatusCode)
    flusher,_:=w.(http.Flusher)
    buffer:=make([]byte,4096)
    for {
        n,e:=resp.Body.Read(buffer)
        if n>0 { if _,err=w.Write(buffer[:n]);err!=nil{return}; if flusher!=nil{flusher.Flush()} }
        if e!=nil{return}
    }
}

func (s *server) chatV2(w http.ResponseWriter,r *http.Request) {
    id:=r.PathValue("id")
    if !s.ownsSession(r,id){fail(w,404,"session not found");return}
    body,err:=io.ReadAll(io.LimitReader(r.Body,18000))
    if err!=nil{fail(w,400,"invalid message");return}
    s.memoryForward(w,r,"POST","/v2/users/"+currentUser(r)+"/sessions/"+id+"/chat",body)
}

func (s *server) closeV2(w http.ResponseWriter,r *http.Request) {
    id:=r.PathValue("id")
    if !s.ownsSession(r,id){fail(w,404,"session not found");return}
    s.memoryForward(w,r,"POST","/v2/users/"+currentUser(r)+"/sessions/"+id+"/close",[]byte("{}"))
}

func (s *server) memoryStateV2(w http.ResponseWriter,r *http.Request) {
    s.memoryForward(w,r,"GET","/v2/users/"+currentUser(r)+"/state",nil)
}

func (s *server) preferencesV2(w http.ResponseWriter,r *http.Request) {
    s.memoryForward(w,r,"GET","/v2/users/"+currentUser(r)+"/preferences",nil)
}

func (s *server) confirmationV2(w http.ResponseWriter,r *http.Request) {
    body,_:=io.ReadAll(io.LimitReader(r.Body,2048))
    s.memoryForward(w,r,"POST","/v2/users/"+currentUser(r)+"/confirmations/"+url.PathEscape(r.PathValue("cid")),body)
}

func (s *server) removePreferenceV2(w http.ResponseWriter,r *http.Request) {
    body,_:=io.ReadAll(io.LimitReader(r.Body,4096))
    s.memoryForward(w,r,"POST","/v2/users/"+currentUser(r)+"/preferences/remove",body)
}

func (s *server) retryMemoryV2(w http.ResponseWriter,r *http.Request) {
    s.memoryForward(w,r,"POST","/v2/users/"+currentUser(r)+"/jobs/"+url.PathEscape(r.PathValue("jid"))+"/retry",[]byte("{}"))
}

func (s *server) updateMe(w http.ResponseWriter,r *http.Request) {
    var input struct{ DisplayName string `json:"display_name"` }
    if json.NewDecoder(io.LimitReader(r.Body,1024)).Decode(&input)!=nil || len(input.DisplayName)<1 || len(input.DisplayName)>60 {
        fail(w,400,"display name required (max 60 bytes)");return
    }
    if _,err:=s.db.Exec(r.Context(),`UPDATE app_users SET display_name=$1 WHERE id=$2`,input.DisplayName,currentUser(r));err!=nil{
        fail(w,500,"could not update account");return
    }
    s.me(w,r)
}

func (s *server) deleteMe(w http.ResponseWriter,r *http.Request) {
    s.memoryForward(w,r,"DELETE","/v2/users/"+currentUser(r),nil)
}
