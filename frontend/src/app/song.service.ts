import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class SongService {

  constructor(private http: HttpClient) { }

  uploadSong(formData: FormData) {
    // Create headers (not strictly necessary, but can help)
    const headers = new HttpHeaders({
      'Accept': 'application/json',
      'Content-Type': 'multipart/form-data',
    });

    return this.http.post<any>('http://localhost:5000/predict', formData, { headers });
  }
}
