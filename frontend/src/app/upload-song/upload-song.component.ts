import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { SongService } from '../song.service';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-upload-song',
  standalone: true,
  imports: [FormsModule, HttpClientModule],
  templateUrl: './upload-song.component.html',
  styleUrl: './upload-song.component.scss',
})
export class UploadSongComponent {
  selectedFile: File | null = null;
  loading: boolean = false;
  genre: string | null = null;
  selectedModel: string = 'svm'; // Default to SVM model

  onFileChange(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
    }
  }

  // Handle model selection (SVM or VGG)
  onModelChange(model: string) {
    this.selectedModel = model;
  }

  constructor(private songService: SongService) {}

  onUpload() {
    if (this.selectedFile) {
      this.loading = true;
      this.genre = null;

      const formData = new FormData();
      formData.append('songFile', this.selectedFile);
      formData.append('model', this.selectedModel); // Add the selected model to the form data

      this.songService.uploadSong(formData).subscribe(
        (response) => {
          this.loading = false;
          this.genre = response.genre;
        },
        (error) => {
          this.loading = false;
          console.error('Error uploading song:', error);
        }
      );
    }
  }
}
