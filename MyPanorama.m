
%% 1)Panorama Main Function%
function [pano] = MyPanorama()

   images = dir("/Users/iwu02/Documents/MATLAB/P2/Images/Set1/*.jpg");
    % Image 1
   [xbest1, ybest1] = ANMS(1);
   image_path=fullfile(images(1).folder, images(1).name);
   gray_image1 = rgb2gray(imread(image_path));
   fd1 = getFeatures(ybest1, xbest1, gray_image1);
   % Image 2
   [xbest2, ybest2] = ANMS(2);
   image_path=fullfile(images(2).folder, images(2).name);
   gray_image2 = rgb2gray(imread(image_path));
   fd2 = getFeatures(ybest2, xbest2, gray_image2);
  
   % Get matches of Image 1 and Image 2
   [matches1, matches2] = getMatches(fd1, fd2, xbest1, xbest2, ybest1, ybest2);
   figure()
   ax = axes;
   showMatchedFeatures(gray_image1, gray_image2, matches1, matches2, "montag", Parent=ax)
   title(ax,"Candidate point matches");
   legend(ax,"Matched points 1","Matched points 2");
   [m1, m2, b1] = RANSAC(matches1, matches2);
   figure()
   ax2 = axes;
   showMatchedFeatures(gray_image1, gray_image2, m1,  m2, "montag", Parent=ax2)
   title(ax2,"After RANSAC");
   legend(ax2,"Matched points 1","Matched points 2");
end

%% 2) ANMS
   function [xbest, ybest] = ANMS(imageX)
       images = dir("/Users/coreywang/Documents/MATLAB/P2/Images/Set1/*.jpg");
       image_path=fullfile(images(imageX).folder, images(imageX).name);
       gray_image = rgb2gray(imread(image_path));
       Nbest = 500;
       xbest = zeros(Nbest);
       ybest = zeros(Nbest);
       image_tensor = cat(3, [], gray_image);
       corners = cornermetric(image_tensor(:,:,1));
       localMax = imregionalmax(corners);
       Nstrong= sum(localMax(:)==1);
       [y x] = find(localMax);
      
       anms = [];
       for i = 1:Nstrong
           anms(i).r = inf;
           anms(i).x = 0;
           anms(i).y = 0;
       end
  
       ED = 0;
       for i = 1:Nstrong
           for j = 1:Nstrong
               if (corners(y(j),x(j)) > corners(y(i),x(i)))
                   % Euclidean Distance
                   ED = (x(i)-x(j))^2 + (y(i)-y(j))^2;
               end
               if(ED < anms(i).r)
                   anms(i).r = ED;
                   anms(i).x = x(j);
                   anms(i).y = y(j);
               end
           end
       end
  
       cells = struct2cell(anms);
       sortvals = cells(1,1,:);
       mat = (cell2mat(sortvals));
       [~,ix] = sort(mat,'descend');
       anms_sorted = anms(ix);
      
       % Selecting Nbest elements
       for i = 1:Nbest
           xbest(i) = anms_sorted(i).x;
           ybest(i) = anms_sorted(i).y;
       end
      
       % Image with Nbest corners plotted
       figure()
       title('ANMS Result');
       imshow(imread(image_path));
       hold on
       plot(xbest(:),ybest(:),'r.');
       hold off
   end
   %% 3) Feature Descriptor
   function [feature_vec] = getFeatures(yBest,xBest,img)
       filter = fspecial('gaussian');
       pad = padarray(img,[20,20],'both');
       for i = 1:length(xBest)
           patch = pad(yBest(i):yBest(i) + 40, xBest(i):xBest(i) + 40);
           blur = imfilter(patch, filter);
           sub_sample = imresize(blur, [8, 8]);
           sub_sample = reshape(sub_sample,[1, 64]);
           sub_sample = double(sub_sample);
           feature_vec(i,:) = (sub_sample - mean(sub_sample))/std(sub_sample);
       end
   end

      %% 4) Feature Matching
   function [matches1, matches2, best_matches] = getMatches(feat_vec1, feat_vec2, xBest1, xBest2, yBest1, yBest2)
       best_matches = [];
       [~, Nbest] = size(xBest1);
       % Looping through the feature vectors of image 1
       for i = 1:Nbest
           [best_j, second_best_j, best_ssd, second_best_ssd] = deal(-1);
           % Looping through the feature vectors of image 2
           for j = 1:Nbest
               difference = feat_vec1(i, :) - feat_vec2(j, :);
               ssd = sum(difference(:).^2);
               if or(best_j == -1, ssd < best_ssd)
                   second_best_ssd = best_ssd;
                   second_best_j = best_j;
                   best_ssd = ssd;
                   best_j = j;
               elseif or(second_best_j == -1, ssd < second_best_ssd)
                   second_best_ssd = ssd;
                   second_best_j = j;
               end
           end
      
           % We want the distances to be far apart
           % We have our best j for this one, compare to second best
           if second_best_ssd - best_ssd > 0.5
               v = reshape([i best_j (second_best_ssd - best_ssd)], [3 1]);
               best_matches = [best_matches v];
           end
       end
      
       [~, num_best_matches] = size(best_matches);
       matches1 = zeros(num_best_matches, 2);
       matches2 = zeros(num_best_matches, 2);
       best_matches = sortrows(best_matches.',3).';
       % Format
       for index = 1:num_best_matches
           match = best_matches(:, index);
           x1 = xBest1(match(1));
           y1 = yBest1(match(1));
           x2 = xBest2(match(2));
           y2 = yBest2(match(2));
           matches1(index, 1) = x1;
           matches1(index, 2) = y1;
           matches2(index, 1) = x2;
           matches2(index, 2) = y2;
       end
   end
   %% 5) RANSAC
   function [inlierSource,inlierDest,H_inlier] = RANSAC(matches1,matches2)
       inliers = 0;
       for j = 1:500
           count = 1;   
           ind = randperm(size(matches1,1),4);
           H = est_homography(matches2(ind,1),matches2(ind,2),matches1(ind,1),matches1(ind,2));                %   Estimating Homography based on these 4 random points
           for i = 1:size(matches1,1)
               [x,y] = apply_homography(H,matches1(i,1),matches1(i,2));          %   Calculating the new coordinates after applying the homography and then finding inliers
               s = (matches2(i,1)-x).^2 + (matches2(i,2)-y).^2; 
              
               %   Finding the distance between estimated and the actual location
               % s > (1*10^(6))
               if (s > (1*10^(5)))
                   inlierSource(count,:) = [matches1(i,1),matches1(i,2)];
                   inlierDest(count,:) = [matches2(i,1),matches2(i,2)];
                   count = count + 1;
               end
           end
           if (count>inliers)                                                            %   Min. of 4 points is required to calculate homography matrix
               inliers = count;
               if (inliers<4)
                   error(message('Not enough inliers to compute Homography'));
               end
           end
           if (count/size(matches1,1)) >= 0.9                                      %   Break the chain if we obtain 95% of matches as inliers
               disp('Inside break');
               break;
           end
       end
       %     tform = est_homography(inlierDest(:,1),inlierDest(:,2),inlierSource(:,1),inlierSource(:,2));
       %     H_inlier = projective2d(tform);
       H_inlier = fitgeotrans(inlierDest,inlierSource,'projective');                   %   Calculating the
   end


