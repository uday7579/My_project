create database school;
use school;

CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(50),
    gender VARCHAR(10),
    class VARCHAR(10),
    age INT,
    city VARCHAR(50)
);

INSERT INTO Students VALUES
(1,'Aman','Male','10A',15,'Delhi'),
(2,'Priya','Female','10A',15,'Mumbai'),
(3,'Rahul','Male','10B',16,'Pune'),
(4,'Sneha','Female','10B',15,'Delhi'),
(5,'Vikas','Male','9A',14,'Jaipur'),
(6,'Neha','Female','9A',14,'Lucknow'),
(7,'Arjun','Male','9B',14,'Mumbai'),
(8,'Kajal','Female','9B',14,'Pune'),
(9,'Rohit','Male','8A',13,'Delhi'),
(10,'Anjali','Female','8A',13,'Jaipur');

CREATE TABLE Teachers (
    teacher_id INT PRIMARY KEY,
    teacher_name VARCHAR(50),
    subject VARCHAR(50),
    salary INT
);

INSERT INTO Teachers VALUES
(101,'Rajesh','Math',50000),
(102,'Sunita','Science',55000),
(103,'Vivek','English',48000),
(104,'Pooja','History',47000),
(105,'Amit','Computer',60000),
(106,'Renu','Math',52000),
(107,'Karan','Science',53000),
(108,'Nisha','English',49000),
(109,'Deepak','Computer',62000),
(110,'Meena','History',46000);

CREATE TABLE Subjects (
    subject_id INT PRIMARY KEY,
    subject_name VARCHAR(50),
    teacher_id INT,
    FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id)
);

INSERT INTO Subjects VALUES
(1,'Math',101),
(2,'Science',102),
(3,'English',103),
(4,'History',104),
(5,'Computer',105),
(6,'Math Advanced',106),
(7,'Science Lab',107),
(8,'English Grammar',108),
(9,'Programming',109),
(10,'World History',110);

CREATE TABLE Marks (
    mark_id INT PRIMARY KEY,
    student_id INT,
    subject_id INT,
    marks INT,
    exam_type VARCHAR(20),
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (subject_id) REFERENCES Subjects(subject_id)
);

INSERT INTO Marks VALUES
(1,1,1,85,'Final'),
(2,2,1,90,'Final'),
(3,3,2,78,'Final'),
(4,4,2,88,'Final'),
(5,5,3,75,'Final'),
(6,6,3,92,'Final'),
(7,7,4,81,'Final'),
(8,8,4,86,'Final'),
(9,9,5,80,'Final'),
(10,10,5,95,'Final');

select student_name from students where age > 14;
select avg(marks) as highest_marks from marks;
select city, count(*) as total_students from students group by city;
select * from students order by age;
select * from marks order by marks desc limit 5;
select * from teachers order by salary desc;
select s.student_name, m.marks
from students s 
join marks m
on s.student_id = m.student_id;
select s.student_name,sb.subject_name, m.marks
from students s 
join marks m 
on s.student_id = m.student_id
join subjects sb 
on m.subject_id = sb.subject_id;

select s.subject_name, t.teacher_name
from subjects s join teachers t 
on s.teacher_id = t.teacher_id;

select s.student_name, m.marks
from students s join marks m 
on s.student_id = m.student_id
where m.marks > 85;

SELECT sub.subject_name,
       s.student_name,
       m.marks
FROM Marks m
JOIN Students s ON m.student_id = s.student_id
JOIN Subjects sub ON m.subject_id = sub.subject_id
WHERE (m.subject_id, m.marks) IN (
    SELECT subject_id, MAX(marks)
    FROM Marks
    GROUP BY subject_id
);

select s.student_name , s.city , m.marks
from students s join marks m 
on s.student_id = m.student_id;

select s.student_name , m.marks from students s join marks m on s.student_id = m.student_id
where m.marks = ( select max(marks) from marks);

select * from teachers where salary = (select max(salary) from teachers);

select s.student_name , m.marks 
from students s 
join marks m 
on s.student_id = m.student_id
where m.marks > (select avg(marks) from marks );

select subject_id , avg(marks) as avg_marks from marks group by subject_id having avg_marks >80;

select s.class, avg(marks) as avg_marks from students s join marks m 
on s.student_id = m.student_id
group by s.class;

select s.city, avg(marks) as avg_marks from students s join marks m 
on s.student_id = m.student_id
group by s.city;

select s.student_name , sum(marks) as total_marks from students s join marks m 
on s.student_id = m.student_id
group by s.student_name;

select s.student_name , m.marks 
from students s join marks m 
on s.student_id = m.student_id
where m.marks >90;

select subject_id , count(student_id) as total_students from marks 
group by subject_id 
having count(student_id) > 1;

select * from teachers where salary > (select avg(salary) from teachers);

select max(marks) from marks where marks < (select max(marks) from marks);

select student_id , marks, rank() over(order by marks desc ) as rank_no from marks ;

select * from (select student_id, subject_id , marks,
 rank() over(partition by subject_id order by marks desc ) 
 rnk from marks 
)x
where rnk = 1;

select student_id , marks , sum(marks) over(order by student_id ) as commulative_marks
from marks;

select s.student_name , m.marks, (m.marks/100.0)*100 as percentage
from students s join marks m on s.student_id = m.student_id;

select s.student_name , m.marks , sb.subject_name, t.teacher_name , m .exam_type
from students s join marks m 
on s.student_id = m.student_id 
join subjects sb 
on m.subject_id = sb.subject_id
join teachers t on t.teacher_id = sb .teacher_id;

select s.city , avg(marks) as avg_marks from students s join marks m 
on s.student_id = m.student_id
group by s.city order by avg_marks desc limit 1;

select t.teacher_name , avg(m.marks) as avg_marks
from teachers t join subjects sb 
on t.teacher_id = sb.teacher_id
join marks m on sb.subject_id = m.subject_id
group by teacher_name order by avg_marks desc limit 1;

select round(
count(case when marks >80 then 1 end )*100.0 /count(*) ,2 ) as percentage_students from marks;

select s.class , avg(m.marks) as avg_marks from students s join marks m 
on s.student_id = m.student_id
group by s.class
order by avg_marks desc limit 1;

select count(distinct student_id) as total_students ,
avg(marks) as average_marks,
min(marks) as min_marks,
max(marks) as max_marks,
round(count(case when marks>=40 then 1 end)*100.0/count(*),2) as pass_percentage from marks;









