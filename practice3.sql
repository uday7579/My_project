create database music;
use music;
CREATE TABLE Users (
    UserID INT PRIMARY KEY,
    UserName VARCHAR(50),
    Email VARCHAR(100),
    Country VARCHAR(50)
);

INSERT INTO Users VALUES
(1,'Amit','amit@gmail.com','India'),
(2,'Rahul','rahul@gmail.com','India'),
(3,'Priya','priya@gmail.com','India'),
(4,'John','john@gmail.com','USA'),
(5,'Emma','emma@gmail.com','UK'),
(6,'Sophia','sophia@gmail.com','Canada'),
(7,'Liam','liam@gmail.com','Australia'),
(8,'Noah','noah@gmail.com','USA'),
(9,'Olivia','olivia@gmail.com','UK'),
(10,'Ava','ava@gmail.com','India');

CREATE TABLE Artists (
    ArtistID INT PRIMARY KEY,
    ArtistName VARCHAR(50),
    Genre VARCHAR(50)
);

INSERT INTO Artists VALUES
(101,'Arijit Singh','Bollywood'),
(102,'Shreya Ghoshal','Bollywood'),
(103,'Ed Sheeran','Pop'),
(104,'Taylor Swift','Pop'),
(105,'Atif Aslam','Romantic'),
(106,'Imagine Dragons','Rock'),
(107,'KK','Bollywood'),
(108,'Neha Kakkar','Pop'),
(109,'Justin Bieber','Pop'),
(110,'Sonu Nigam','Bollywood');

CREATE TABLE Songs (
    SongID INT PRIMARY KEY,
    SongName VARCHAR(100),
    ArtistID INT,
    Duration INT,
    ReleaseYear INT,
    FOREIGN KEY (ArtistID) REFERENCES Artists(ArtistID)
);

INSERT INTO Songs VALUES
(201,'Kesariya',101,270,2022),
(202,'Tum Hi Ho',101,260,2013),
(203,'Perfect',103,280,2017),
(204,'Love Story',104,240,2008),
(205,'Jeena Jeena',105,250,2015),
(206,'Believer',106,230,2017),
(207,'Zara Sa',107,245,2008),
(208,'Dilbar',108,220,2018),
(209,'Sorry',109,210,2015),
(210,'Abhi Mujh Mein Kahin',110,290,2012);

CREATE TABLE PlayHistory (
    PlayID INT PRIMARY KEY,
    UserID INT,
    SongID INT,
    PlayDate DATE,
    PlayCount INT,
    FOREIGN KEY (UserID) REFERENCES Users(UserID),
    FOREIGN KEY (SongID) REFERENCES Songs(SongID)
);

INSERT INTO PlayHistory VALUES
(1,1,201,'2026-01-01',5),
(2,2,202,'2026-01-02',3),
(3,3,203,'2026-01-03',8),
(4,4,204,'2026-01-04',4),
(5,5,205,'2026-01-05',7),
(6,6,206,'2026-01-06',6),
(7,7,207,'2026-01-07',2),
(8,8,208,'2026-01-08',9),
(9,9,209,'2026-01-09',3),
(10,10,210,'2026-01-10',10);

select * from users;
select * from songs where ReleaseYear > 2015;
select * from artists where genre = 'pop';
select songname,duration from songs;
select * from users where country = 'india';
select s.songname , a.artistname from songs s join artists a 
on s.artistid = a.artistid;
select u.username, s.songname from users u join playhistory ph on u.userid = ph.userid 
join songs s on ph.songid = s.songid;
select s.songname , count(ph.playcount) as total_plays from songs s join playhistory ph on s.songid = ph.songid
group by s.songname;
select s.songname , sum(ph.playcount) as total_plays from songs s join playhistory ph on s.songid = ph.songid
group by s.songname order by total_plays desc limit 5;
select avg(duration) as avg_song_play from songs;
select s.songname , u.username , ph.playcount 
from songs s join playhistory ph on s.songid = ph.songid
join users u on ph.userid = u.userid;
select a.artistname , s.songname from artists a join songs s on a.artistid = s.artistid;
select s.songname , u.username from users u join playhistory ph on u.userid = ph.userid join songs s 
on ph.songid = s.songid where u.username = 'amit';
select distinct u.username from users u join playhistory ph 
on u.userid = ph.userid join songs s on ph.songid = s.songid 
join artists a on s.artistid = s.artistid where artistname = 'arijit singh';
select a.artistname , sum(ph.playcount) as total_plays from artists a join songs s  
on s.artistid = a.artistid join playhistory ph on s.songid = ph.songid
group by a.artistname;
select a.artistname , sum(s.songid) as total_songs from artists a join songs s 
on a.artistid = s.artistid group by a.artistname;
select country , count(*) as total_users from users group by country;
select a.artistname , count(s.songid) as total_songs from artists a join songs s 
on a.artistid = s.artistid group by artistname having count(s.songid) > 1;
select s.songname , count(ph.playcount) as total_plays from songs s join playhistory ph on s.songid = ph.songid 
group by s.songname having sum(ph.playcount) >5;
select genre , count(*) artistcount from artists group by genre having count(*) >2;
select songname from songs where songid = ( select songid  from playhistory group by songid order by sum(playcount) desc limit 1);
select * from songs where duration = ( select max(duration) from songs);
select distinct u.username from users u join playhistory ph 
on u.userid = ph.userid 
where ph.songid = ( select songid from playhistory ph group by songid order by sum(playcount) desc limit 1);
select distinct a.artistname from artists a join songs s on a.artistid = s.artistid join playhistory ph on s.songid = ph.songid 
group by a.artistname having avg(ph.playcount) > ( select avg(playcount) from playhistory );
select s.songname , sum(ph.playcount) as total_plays from songs s join playhistory ph on s.songid = ph.songid group by s.songname
order by total_plays  desc limit 1 offset 1;
select s.songname , sum(ph.playcount) as play_count , rank() over(order by sum(ph.playcount) desc ) as song_rnk
from songs s join playhistory ph on s.songid = ph.songid group by s.songname;
select a.artistname , sum(ph.playcount) as total_plays , rank() over( order by sum(ph.playcount) desc) as rnk from artists a
join songs s on s.artistid = a.artistid join playhistory ph on s.songid = ph.songid group by a.artistname;
select playdate , sum(playcount) as daily_plays , sum(sum(playcount)) over(order by playdate ) as commulative_plays from playhistory 
group by playdate;
with songrank as (select s.songname, sum(ph.playcount) total_plays, dense_rank() over( order by sum(ph.playcount) desc ) rnk from songs 
s join playhistory ph on s.songid = ph.songid group by songname) select * from songrank where rnk <=3;
with usersong as ( select u.username, s.songname, ph.playcount, row_number() over(partition by u.username order by ph.playcount desc ) rnk
from users u join playhistory ph on u.userid = ph.userid join songs s on ph.songid = s.songid ) 
select * from usersong where rnk = 1;