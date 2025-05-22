import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class WordCountMapper extends Mapper<LongWritable, Text, Text, IntWritable>{
    Text k = new Text();
    IntWritable v = new IntWritable();
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        // 1 获取一行
        String line = value.toString();
        // 2 切割
        String[] fields = line.split("\\s+"); // 以空白字符分割
        if(fields.length == 2){
            // 3 获取科目和分数
            String subject = fields[0];
            int score = Integer.parseInt(fields[1]);
            // 4 输出
            k.set(subject);
            v.set(score);
            context.write(k, v);
        }
    }
}
